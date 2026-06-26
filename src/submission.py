import os
import time
import sys
import contextlib
import gc
import csv
import warnings
warnings.filterwarnings('ignore')
import zipfile
import logging
import argparse
import textstat
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from langchain_community.llms import LlamaCpp
from langchain_google_genai import ChatGoogleGenerativeAI
import transformers

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('huggingface_hub').setLevel(logging.ERROR)
transformers.logging.set_verbosity_error()

PROJECT_ROOT = os.path.abspath('.')
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.graph import setup_system, run_simplification
from src.signals import normalize_signals

def setup_models():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("[FAIL] Error: No se ha encontrado la GOOGLE_API_KEY en el archivo .env")
        return None, None

    print("[AI] Conectando con Gemini 2.5 Flash...")
    try:
        gemini_llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', api_key=api_key, temperature=0.1)
    except Exception as e:
        print(f"[FAIL] Error conectando con Gemini: {e}")
        return None, None
    
    import torch
    has_gpu = torch.cuda.is_available()
    gpu_layers = -1 if has_gpu else 0
    n_threads = 4 if has_gpu else 8

    print(f"Hardware detectado: {'GPU NVIDIA' if has_gpu else 'SOLO CPU'}")
    gpu_label = "todas las" if gpu_layers == -1 else f"{gpu_layers} capa{'s' if gpu_layers != 1 else ''}"
    print(f"[LOCAL] Cargando RigoChat GGUF ({n_threads} hilo{'s' if n_threads != 1 else ''}, {gpu_label} GPU)...")
    try:
        model_path = hf_hub_download(
            repo_id="IIC/RigoChat-7b-v2-GGUF",
            filename="rigochat-7b-v2-Q4_K_M.gguf",
            cache_dir="./models"
        )
        with contextlib.redirect_stderr(open(os.devnull, 'w')):
            rigochat_llm = LlamaCpp(
                model_path=model_path,
                n_ctx=2048,
                n_threads=n_threads,
                n_gpu_layers=gpu_layers,
                temperature=0.2,
                verbose=False
            )
        print(f"[OK] RigoChat listo ({'GPU' if has_gpu else 'CPU'})")
    except Exception as e:
        print(f"[FAIL] Error cargando RigoChat: {e}")
        return None, None

    return gemini_llm, rigochat_llm

def create_zip(team_name):
    zip_filename = f"{team_name}.zip"
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(team_name):
            for file in files:
                filepath = os.path.join(root, file)

                arcname = os.path.relpath(filepath, team_name)
                z.write(filepath, arcname)
    print(f"[OK] ZIP creado: {zip_filename}")

def merge_csv_progress(main_file, secondary_file):
    if not os.path.exists(secondary_file):
        return
    if not os.path.exists(main_file):
        os.rename(secondary_file, main_file)
        return
    
    try:
        with open(secondary_file, 'r', encoding='utf-8') as f_in, \
             open(main_file, 'a', encoding='utf-8', newline='') as f_out:
            reader = csv.reader(f_in)
            next(reader, None) 
            writer = csv.writer(f_out)
            for row in reader:
                writer.writerow(row)
        os.remove(secondary_file)
    except Exception as e:
        print(f"[WARNING] Error al fusionar archivos de progreso: {e}")

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Generador de Entrega IberLEF 2026')
    parser.add_argument('--input', type=str, default='data/es_test_data.csv', help='Ruta al CSV de test')
    parser.add_argument('--team', type=str, default='HULAT-UC3M', help='Nombre del equipo')
    parser.add_argument('--run', type=str, default='RUN1', help='Nombre del run (RUN1, RUN2...)')
    parser.add_argument('--lexical', type=str, choices=['on', 'off', 'auto'], default='auto', 
                        help='Activar/desactivar el Agente Léxico (auto elige según el RUN)')
    parser.add_argument('--subset', type=int, default=None, help='Procesar solo los primeros N ejemplos (para test)')
    parser.add_argument('--resume', action='store_true', help='Continuar desde el último punto guardado')
    args = parser.parse_args()

    # Configuración del Agente Léxico
    if args.lexical == 'auto':
        is_run1 = "RUN1" in args.run.upper()
        os.environ["DISABLE_LEXICAL_AGENT"] = "true" if is_run1 else "false"
    else:
        os.environ["DISABLE_LEXICAL_AGENT"] = "true" if args.lexical == 'off' else "false"

    status_lex = "DESACTIVADO" if os.environ["DISABLE_LEXICAL_AGENT"] == "true" else "ACTIVADO"
    print(f"[CONFIG] Modo {args.run} | Agente Léxico: {status_lex}")


    out_dir = os.path.join(args.team, 'ES')
    os.makedirs(out_dir, exist_ok=True)
    
    out_file = os.path.join(out_dir, f"{args.run}.csv")
    temp_file = out_file + ".partial"
    detailed_file = f"test_results_detailed_{args.run}.csv"
    detailed_temp = detailed_file + ".partial"

    gemini, rigochat = setup_models()
    if not gemini or not rigochat:
        return

    app, glossary, lexical_resources = setup_system(gemini, rigochat)

    test_data = []
    try:
        with open(args.input, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                test_data.append(row)
    except Exception as e:
        print(f"[FAIL] Error leyendo CSV: {e}")
        return

    if args.subset:
        test_data = test_data[:args.subset]
        print(f"[WARNING]  MODO TEST: Procesando solo {args.subset} ejemplo{'s' if args.subset != 1 else ''} para validación rápida.")

    print(f"\n" + "="*60)
    print(f"[START] INICIANDO PROCESAMIENTO: {len(test_data)} instancias")
    print("="*60 + "\n")

    # Resume logic
    processed_ids = set()
    resume_mode = args.resume
    
    progress_exists = os.path.exists(temp_file) or os.path.exists(out_file)

    if progress_exists:
        # Contar filas existentes
        count = 0
        file_to_check = temp_file if os.path.exists(temp_file) else out_file
        try:
            with open(file_to_check, 'r', encoding='utf-8') as f:
                count = sum(1 for _ in csv.DictReader(f))
        except: pass

        if not args.resume:
            print(f"\n[WARNING]  ATENCIÓN: Se ha detectado progreso previo ({count} ejemplo{'s' if count != 1 else ''} encontrado{'s' if count != 1 else ''}).")
            while True:
                choice = input(f"   ¿Deseas [R]eanudar (mantener datos), [S]obrescribir (borrar todo) o [Q]uitar/Salir? (R/S/Q): ").strip().lower()
                if choice == 'r':
                    resume_mode = True
                    break
                elif choice == 's':
                    print("Limpiando y empezando de cero...")
                    for f in [temp_file, out_file, detailed_temp, detailed_file]:
                        if os.path.exists(f): os.remove(f)
                    resume_mode = False
                    break
                elif choice == 'q':
                    print("[EXIT] Saliendo sin realizar cambios para proteger tus datos.")
                    return
                else:
                    print("[FAIL] Opción no válida. Por favor, elige R, S o Q.")
        
    if resume_mode:
        merge_csv_progress(temp_file, out_file)
        merge_csv_progress(detailed_temp, detailed_file)
        
        if os.path.exists(temp_file):
            with open(temp_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    processed_ids.add((row['document_id'], row['original_sentence_id']))
            print(f"[RETRY] Modo REANUDAR activo: {len(processed_ids)} ejemplo{'s' if len(processed_ids) != 1 else ''} ya cargado{'s' if len(processed_ids) != 1 else ''}.")

    # Bucle de procesamiento
    fieldnames_official = ['document_id', 'original_sentence_id', 'system']
    fieldnames_detailed = [
        'id', 'original', 'output', 'strategy', 'decision', 
        'meaningbert', 'legibilidad', 'longitud_frase', 
        'sintaxis_clara', 'sencillez_lexica', 'tiempo_seg',
        'lexical_agent'
    ]
    
    mode = 'a' if resume_mode and os.path.exists(temp_file) else 'w'
    
    with open(temp_file, mode, encoding='utf-8', newline='') as f_off, \
         open(detailed_temp, mode, encoding='utf-8', newline='') as f_det:
        
        writer_off = csv.DictWriter(f_off, fieldnames=fieldnames_official)
        writer_det = csv.DictWriter(f_det, fieldnames=fieldnames_detailed)
        
        if mode == 'w':
            writer_off.writeheader()
            writer_det.writeheader()

        for i, row in enumerate(test_data):
            doc_id = row['document_id']
            sent_id = row['original_sentence_id']
            original = row['original_text']
            
            if (doc_id, sent_id) in processed_ids:
                continue

            print(f"* [{i+1}/{len(test_data)}] Procesando {doc_id} (ID {sent_id})")
            print(f"   Fuente: {original[:70]}..." if len(original) > 70 else f"   Fuente: {original}")
            
            try:
                start_iter = time.time()
                state = run_simplification(app, glossary, lexical_resources, gemini, original, "")
                iter_time = time.time() - start_iter
                
                output = state.get('final_output', '').strip()
                meaningbert = state.get('final_meaningbert', 0.0)
                
                # Obtener y normalizar señales finales
                raw_signals = state.get('final_signals', {})
                norm = normalize_signals(raw_signals)
                legibilidad_norm = round(max(0.0, min(100.0, textstat.fernandez_huerta(output))), 3) if output else 0
                
                writer_off.writerow({
                    'document_id': doc_id, 'original_sentence_id': sent_id, 'system': output
                })
                writer_det.writerow({
                    'id': f"{doc_id}_{sent_id}", 
                    'original': original, 
                    'output': output,
                    'strategy': state.get('strategy', 'N/A'), 
                    'decision': state.get('final_decision', 'N/A'),
                    'meaningbert': round(meaningbert, 3), 
                    'legibilidad': legibilidad_norm,
                    'longitud_frase': norm.get('longitud_frase', 0),
                    'sintaxis_clara': norm.get('sintaxis_clara', 0),
                    'sencillez_lexica': norm.get('sencillez_lexica', 0),
                    'tiempo_seg': round(iter_time, 2),
                    'lexical_agent': 'SÍ' if state.get('lexical_agent_applied') else 'NO'
                })
                
                f_off.flush()
                f_det.flush()
                print(f"   [OK] ÉXITO | MeaningBERT: {round(meaningbert,2)} | Legibilidad (Huerta): {legibilidad_norm} | Tiempo: {round(iter_time,1)}s\n")
            except Exception as e:
                print(f" [FAIL] Error: {e}")
            
            gc.collect()

    # Finalizar
    os.rename(temp_file, out_file)
    os.rename(detailed_temp, detailed_file)
    
    total_segundos = 0
    try:
        with open(detailed_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_segundos += float(row.get('tiempo_seg', 0))
    except:
        total_segundos = time.time() - start_time

    horas = int(total_segundos // 3600)
    minutos = int((total_segundos % 3600) // 60)
    segundos = int(total_segundos % 60)
    
    print(f"\n[PROFIT] Procesamiento finalizado. Tiempo total acumulado: {horas}h {minutos}m {segundos}s.")
    print(f"   -> Archivo oficial:   {out_file}")
    print(f"   -> Archivo detallado: {detailed_file}")
    
    create_zip(args.team)

if __name__ == "__main__":
    main()
