import torch, random, time
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pathlib

CLASS_NAMES = ["NOMINAL", "CAUTION", "WARNING"]
CLASS_COLORS = {0: (0, 220, 80), 1: (220, 160, 0), 2: (220, 40, 40)}

def generate_hard_image(cid):
    # Create a completely random background
    bg_color = (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100))
    img = Image.new("RGB", (640, 360), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Add random noise lines
    for _ in range(20):
        draw.line([(random.randint(0, 640), random.randint(0, 360)), 
                   (random.randint(0, 640), random.randint(0, 360))], 
                  fill=(random.randint(0,255), random.randint(0,255), random.randint(0,255)), width=random.randint(1,3))
    
    # Add the WCA box somewhere in the image (not fixed position)
    box_w, box_h = random.randint(100, 200), random.randint(30, 80)
    x1 = random.randint(0, 640 - box_w)
    y1 = random.randint(0, 360 - box_h)
    
    # Base color for class but with severe variation
    base_r, base_g, base_b = CLASS_COLORS[cid]
    r = max(0, min(255, base_r + random.randint(-60, 60)))
    g = max(0, min(255, base_g + random.randint(-60, 60)))
    b = max(0, min(255, base_b + random.randint(-60, 60)))
    
    draw.rectangle([x1, y1, x1+box_w, y1+box_h], fill=(r,g,b), outline=(255,255,255), width=2)
    
    # Add text
    texts = {0: ["SYS OK", "NOMINAL", "READY"], 1: ["LOW PRESS", "CAUTION", "CHK FUEL"], 2: ["FIRE", "ENGINE FAIL", "WARNING"]}
    draw.text((x1+10, y1+10), random.choice(texts[cid]), fill=(0,0,0))
    
    # Save temp
    path = f"temp_hard_{random.randint(1000,9999)}.png"
    img.save(path)
    return path

def run():
    import sys
    sys.path.append('.')
    from ml_trainer_v3 import predict
    
    print("\n  🔍 GERÇEKÇİ (UNSEEN DATA) BENCHMARK BAŞLIYOR...")
    print("  Model daha önce HİÇ GÖRMEDİĞİ dinamik arkaplanlı, rastgele pozisyonlu")
    print("  ve yüksek gürültülü uçuş paneli görüntüleriyle test ediliyor...\n")
    
    correct = 0
    total = 150
    times = []
    
    for i in range(total):
        cid = random.randint(0, 2)
        img_path = generate_hard_image(cid)
        
        r = predict(img_path)
        pathlib.Path(img_path).unlink(missing_ok=True)
        
        if "error" in r: continue
        times.append(r["ms"])
        if CLASS_NAMES.index(r["class"]) == cid:
            correct += 1
            
        if (i+1) % 30 == 0:
            print(f"  [{i+1}/{total}] Doğruluk: {correct/(i+1)*100:.1f}%")
            
    acc = correct / total * 100
    avg = sum(times) / len(times) if times else 0
    print(f"\n  🎯 GERÇEK ACCURACY TEST SONUCU")
    print(f"  Toplam Test: {total} yepyeni görüntü")
    print(f"  Accuracy   : {acc:.1f}%")
    print(f"  Avg Time   : {avg:.1f} ms")

if __name__ == "__main__":
    run()
