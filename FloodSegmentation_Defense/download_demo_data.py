"""Демонстрацияға арналған Sen1Floods11 суреттерін ресми көзден жүктеу.

Деректер репозиторийге қосылмаған: оларды қайта тарату шарттары толық анықталмаған.
Қолдану: python download_demo_data.py
"""
from pathlib import Path
import requests

BASE = "https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/"
CHIPS = ["Bolivia_103757", "Bolivia_129334"]
ROOT = Path(__file__).parent / "data"

for chip in CHIPS:
    for folder in ["S1Hand", "LabelHand"]:
        path = ROOT / folder / f"{chip}_{folder}.tif"
        if path.exists() and path.stat().st_size:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(BASE + f"{folder}/{path.name}", timeout=(20, 120))
        response.raise_for_status()
        path.write_bytes(response.content)
        print("Жүктелді:", path)
print("Демонстрация деректері дайын:", ROOT)
