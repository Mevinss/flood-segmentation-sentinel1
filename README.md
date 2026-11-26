# Спутниктік суреттерден су басқан аумақтарды сегментациялау

Курстық жұмыс: **«Терең оқыту негізінде спутниктік суреттерден су басқан аумақтарды анықтау және сегментациялау»**
Пән: Терең оқыту (TO 4304) · Л.Н. Гумилев атындағы ЕҰУ · Серикпаева Аружан, ЖИТ-31

**Sen1Floods11 · Sentinel-1 VV/VH · U-Net / ResNet-34 / Attention U-Net · PyTorch · Google Colab**

## Нәтижелер

Деректер: Sen1Floods11 v1.1 жинағының қолмен белгіленген 446 фрагменті, ресми бөлу бойынша: train 252, validation 89, test 90 және бөлек Боливия 15.
Метрикалар тест жиынтығының барлық жарамды пиксельдері бойынша есептелген (су класы, %).

| Әдіс | Тест IoU | Тест F1 | Боливия IoU | Боливия F1 |
|---|---|---|---|---|
| Otsu (VH шегі) | 23,14 | 37,58 | 40,44 | 57,59 |
| Қарапайым U-Net | 66,82 | 80,11 | 58,95 | 74,17 |
| ResNet-34 U-Net, нөлден | 68,22 | 81,11 | 63,54 | 77,71 |
| ResNet-34 U-Net, ImageNet | 68,46 | 81,28 | 68,52 | 81,32 |
| **Attention U-Net, ResNet-34** | **68,86** | **81,56** | **69,55** | **82,04** |

Шектеулер: модельдер бір seed-пен (42) оқытылған. Валидацияда толықтығы теңестірілген нүктеде Attention U-Net-тің IoU мәні базалық ResNet-34 U-Net-тен төмен (67,96% және 69,24%), сондықтан тесттегі 0,40 пайыздық тармақ айырманы тұрақты жақсару деп санауға болмайды.

## Репозиторий құрылымы

| Бума | Мазмұны |
|---|---|
| `flood/` | Негізгі код: деректерді жүктеу және аудит (`data.py`), модельдер (`models.py`: U-Net, ResNet-34 U-Net, Attention U-Net), оқыту (`train.py`), бағалау және карталар (`evaluate.py`), метрикалар (`core.py`), жақсартулар (`improve.py`) |
| `tests/` | Автоматты тексерулер: NoData, бөлу, метрикалар, нормализация, геосілтеме, оқытуды жалғастыру, жаңа шығын функциясы |
| `notebooks/` | Colab ноутбуктері: `FloodSegmentation_Colab.ipynb` (Otsu, U-Net, ResNet-34), `Attention_Bolivia_Colab.ipynb` (Attention U-Net), `01–03` жеке кезеңдер, `FloodSegmentation_Improve.ipynb` (жақсартулар мен бірнеше seed; әлі іске қосылмаған) |
| `reports/` | Тәжірибе скрипттері, оқыту тарихы, салыстыру кестелері, Боливия бойынша қателер талдауы және суреттер |
| `FloodSegmentation_Defense/` | **Дайын оқытылған модель** (`models/attention_unet_resnet34.joblib`), `inference.py`, демонстрация ноутбугі `FloodSegmentation_KZ.ipynb`, толық нұсқаулық `README_KZ.md` |

## Дайын модельді қолдану

```bash
cd FloodSegmentation_Defense
python -m pip install -r requirements.txt
python download_demo_data.py
python inference.py --model models/attention_unet_resnet34.joblib \
  --image data/S1Hand/Bolivia_103757_S1Hand.tif --output outputs/prediction.tif
```

Кіріс — екі арналы 512×512 Sentinel-1 GeoTIFF (VV, VH, dB). Шығыс растрда 0 — құрлық, 1 — су, 255 — жарамсыз пиксель.
Модель: Attention U-Net, ResNet-34 энкодері, 34-дәуір, шек 0,473. Экспортталған салмақтар бастапқы checkpoint-пен дәл сәйкес келетіні тексерілген (`outputs/checkpoint_export_verification.json`).
Толығырақ: [`FloodSegmentation_Defense/README_KZ.md`](FloodSegmentation_Defense/README_KZ.md).

## Тәжірибені қайталау (Colab)

1. `notebooks/FloodSegmentation_Colab.ipynb` ноутбугін Colab-та ашып, T4 GPU таңдаңыз.
2. Ұяшықтарды ретімен орындаңыз. Деректер ресми көзден `/content/flood_project/data` бумасына жүктеледі, нәтижелер `MyDrive/FloodSegmentation` бумасына сақталады.
3. Attention U-Net: `notebooks/Attention_Bolivia_Colab.ipynb`. Негізгі тестті қайта есептеу: `reports/attention_main_test_colab.py`.
4. Сеанс үзілсе, оқыту `last.pt` арқылы соңғы дәуірден жалғасады.

Негізгі баптаулар: AdamW, lr = 1e-4, weight decay = 1e-4, batch = 8, ең көбі 100 дәуір, ерте тоқтату 15 дәуір, seed = 42, шығын BCE + Dice. Нормализация тек train бөлігі бойынша есептеледі, шек тек validation бойынша таңдалады.

Тесттер: `python -m unittest discover -s tests -v`

## Деректер

Ресми көзі: [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) (Bonafilia et al., CVPR Workshops 2020). Суреттер репозиторийге қосылмаған, код оларды бастапқы көзден жүктейді. Деректердегі су маскасына тұрақты су да кіреді, сондықтан JRC картасын алып тастағаннан кейінгі карта уақытша су басудың тәуелсіз расталған бағасы емес.

## ЖИ құралдары

Кодты реттеу мен экспорттауда OpenAI Codex, ал есепті тексеру, әдебиет іздеу және жақсартулар ноутбугін дайындауда Claude (Anthropic) қолданылды. Барлық сандық нәтижелер Google Colab-тағы нақты есептеулерден алынды.
