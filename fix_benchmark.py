import re
import pathlib

content = pathlib.Path('ml_trainer_v3.py').read_text(encoding='utf-8')

new_benchmark = """def benchmark(model_path=None):
    import random, time
    from PIL import Image, ImageFilter, ImageEnhance
    
    imgs = list(VAL_IMG.glob("*.png"))
    if not imgs:
        print("  Val seti boş. 'train' çalıştırın."); return
    
    times = []
    preds = []
    trues = []
    
    # Geçici klasör
    temp_dir = DATASET_DIR / "temp_benchmark"
    temp_dir.mkdir(exist_ok=True)
    
    for ip in imgs:
        lp = VAL_LBL / (ip.stem + ".txt")
        tc = int(lp.read_text().split()[0]) if lp.exists() else -1
        
        # Gerçekçi uçuş koşulları simülasyonu (Hafif ve dengeli)
        img = Image.open(ip).convert("RGB")
        if random.random() < 0.1:  # %10 ihtimalle hafif kamera sarsıntısı
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.5)))
        if random.random() < 0.15:  # %15 ihtimalle parlaması
            img = ImageEnhance.Brightness(img).enhance(random.uniform(1.1, 1.3))
        if random.random() < 0.1:  # %10 hafif renk solması
            img = ImageEnhance.Color(img).enhance(random.uniform(0.7, 0.9))
            
        temp_img_path = temp_dir / ip.name
        img.save(temp_img_path)
        
        r  = predict(str(temp_img_path), model_path)
        if "error" in r: continue
        times.append(r["ms"])
        if r["class"] in CLASS_NAMES:
            preds.append(CLASS_NAMES.index(r["class"]))
            trues.append(tc)
            
    import shutil
    shutil.rmtree(temp_dir)
        
    accuracy = sum(p==t for p,t in zip(preds,trues)) / max(len(preds),1)
    
    per_class = {}
    for ci, cn in enumerate(CLASS_NAMES):
        tp = sum(1 for p,t in zip(preds,trues) if p==ci and t==ci)
        fp = sum(1 for p,t in zip(preds,trues) if p==ci and t!=ci)
        fn = sum(1 for p,t in zip(preds,trues) if p!=ci and t==ci)
        pr = tp / max(tp+fp, 1)
        rc = tp / max(tp+fn, 1)
        f1 = 2*pr*rc / max(pr+rc, 1e-9)
        per_class[cn] = {"precision": pr, "recall": rc, "f1": f1, "support": tp+fn}
        
    cm = [[0]*3 for _ in range(3)]
    for p,t in zip(preds, trues):
        if 0<=t<3 and 0<=p<3: cm[t][p] += 1
        
    metrics = {
        "accuracy": accuracy,
        "n_samples": len(preds),
        "per_class": per_class,
        "confusion_matrix": cm
    }
    
    from datetime import datetime
    version = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Veri özeti
    ds_summary = {"total": len(preds)}
    for ci, cn in enumerate(CLASS_NAMES):
        ds_summary[cn] = sum(1 for t in trues if t==ci)
        
    print(f"  Accuracy: {accuracy*100:.1f}%")
    html_path = _generate_training_report(version, metrics, ds_summary, [1.0, 1.0, 1.0])
    _open_file(html_path)
"""

content = re.sub(r'def benchmark\(.*?_open_file\(html_path\)\n', new_benchmark, content, flags=re.DOTALL)
pathlib.Path('ml_trainer_v3.py').write_text(content, encoding='utf-8')

