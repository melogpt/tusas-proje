import sys, pathlib, shutil, random
from PIL import Image, ImageDraw
import numpy as np

CLASS_NAMES = ["NOMINAL", "CAUTION", "WARNING"]
CLASS_COLORS = {0: (0, 220, 80), 1: (220, 160, 0), 2: (220, 40, 40)}
DATASET_DIR = pathlib.Path("ml_dataset")
DATASET_DIR.joinpath("images", "all").mkdir(parents=True, exist_ok=True)
DATASET_DIR.joinpath("labels", "all").mkdir(parents=True, exist_ok=True)

# Generate 500 extremely varied synthetic images per class
for cid in range(3):
    print(f"Generating 500 diverse samples for {CLASS_NAMES[cid]}...")
    for i in range(500):
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
        
        fname = f"SYNTH_DIVERSE_{cid}_{i}.png"
        ipath = DATASET_DIR / "images" / "all" / fname
        lpath = DATASET_DIR / "labels" / "all" / (fname.replace(".png", ".txt"))
        img.save(ipath)
        lpath.write_text(f"{cid} 0.5 0.5 1.0 1.0\n")

print("Added 1500 diverse samples.")
