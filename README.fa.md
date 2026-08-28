# پیش‌بینی دمای روز آینده با استفاده از یادگیری ماشین و مهندسی ویژگی‌ها

## هدف پروژه

طراحی یک مدل یادگیری ماشین برای پیش‌بینی دمای میانگین روز آینده در تهران با استفاده از داده‌های هواشناسی روزانه و ویژگی‌های مهندسی‌شده.

## داده‌ها

- موقعیت: تهران، ایران (عرض جغرافیایی: 35.6892، طول جغرافیایی: 51.3134)  
- بازه زمانی: 2020-01-01 تا 2025-12-31  
- تعداد رکوردها: 2192 روز  
- منبع: Open-Meteo Historical Weather API  
- متغیر هدف: دمای میانگین روز بعد (`tavg` فردا)

### دریافت داده‌ها

داده‌های خام از API آرشیو Open-Meteo و با استفاده از اسکریپت `src/download_data.py` دانلود می‌شوند. این اسکریپت داده‌ها را مستقیماً از سرور Open-Meteo دریافت و در پوشهٔ `data` پروژه ذخیره می‌کند.

برای دانلود داده‌ها، از ریشهٔ پروژه اجرا کنید:

```bash
python src/download_data.py
```

پس از اجرا، فایل `data/weather.csv` در پوشهٔ پروژه ایجاد می‌شود.

پارامترهای کلیدی درخواست:

- `latitude=35.6892`, `longitude=51.3134`: مختصات تهران  
- `start_date=2020-01-01`, `end_date=2025-12-31`: بازهٔ زمانی داده‌ها  
- `daily`: متغیرهای روزانه شامل دمای میانگین، کمینه و بیشینه، بارش، برف، جهت و سرعت باد، فشار، رطوبت و تابش خورشید  
- `timezone=Asia/Tehran`: منطقهٔ زمانی  
- `format=csv`: قالب خروجی

## مهندسی ویژگی

از داده‌های خام، 46 ویژگی جدید ساخته شده است، از جمله:

- ویژگی‌های تأخیری دما (1، 2، 3، 7، 14 روز قبل)  
- میانگین و انحراف معیار متحرک دما در پنجره‌های 3، 7، 14 و 30 روزه  
- تغییرات کوتاه‌مدت دما، فشار و باد  
- دامنه دما، نقطه میانی دما و انحراف از میانگین 30 روزه  
- ویژگی‌های فصلی و زمانی (ماه، روز سال، سینوسی/کسینوسی روز سال، زمستان/تابستان/دوره گذار)  
- تعامل دما با فشار، باد و بارش  
- ویژگی‌های نوسان‌پذیری، روند کوتاه‌مدت و پرچم‌های دمای غیرعادی

پس از مهندسی ویژگی، مجموعه داده شامل 46 ویژگی و یک متغیر هدف است.

## مدل‌ها و نتایج

مدل‌های مقایسه‌شده:

- Baseline: دمای فردا = دمای امروز  
- Linear Regression  
- Ridge  
- Random Forest  
- Gradient Boosting  
- XGBoost  

بهترین مدل بر اساس اعتبارسنجی سال 2024، Linear Regression بود.

### نتایج آزمون نهایی (سال 2025)

| معیار | Baseline | Linear Regression |
|---|---:|---:|
| MAE | 1.196 | 0.975 |
| RMSE | 1.579 | 1.248 |

بهبود MAE نسبت به Baseline: حدود 18.5 درصد.

## ساختار پروژه

```text
project/
├── data/
│   └── weather.csv
├── models/
│   ├── best_temperature_model.joblib
│   └── model_features.joblib
├── outputs/
│   ├── model_results_2025.csv
│   ├── feature_importance_2025.csv
│   ├── high_error_days_model_comparison.csv
│   ├── tehran_2025_prediction_vs_actual.png
│   ├── tehran_2024_validation_comparison.png
│   └── tehran_feature_importance.png
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── train_temperature_model.py
│   ├── predict_for_date.py
│   ├── analyze_errors.py
│   └── download_data.py
├── README.md
├── README.fa.md
└── requirements.txt
```

## راهنمای اجرا

### 1) نصب وابستگی‌ها

کتابخانه‌های مورد نیاز را نصب کنید:

```bash
pip install -r requirements.txt
```

فایل `requirements.txt` شامل بسته‌های زیر است:

```text
pandas
numpy
scikit-learn
xgboost
matplotlib
joblib
```

### 2) دانلود داده‌ها

برای دانلود خودکار داده‌ها از Open-Meteo، از ریشهٔ پروژه اجرا کنید:

```bash
python src/download_data.py
```

خروجی این دستور، فایل `data/weather.csv` در پوشهٔ پروژه است.

### 3) آموزش مدل و تولید خروجی‌ها

```bash
python src/train_temperature_model.py
```

**خروجی‌ها:**

- `models/best_temperature_model.joblib`: مدل نهایی Linear Regression  
- `models/model_features.joblib`: لیست ویژگی‌های استفاده‌شده  
- `outputs/model_results_2025.csv`: پیش‌بینی‌ها و خطاهای سال 2025  
- `outputs/feature_importance_2025.csv`: اهمیت ویژگی‌ها  
- `outputs/tehran_2025_prediction_vs_actual.png`: نمودار پیش‌بینی در مقابل واقعیت  
- `outputs/tehran_2024_validation_comparison.png`: مقایسه مدل‌ها در اعتبارسنجی  
- `outputs/tehran_feature_importance.png`: نمودار اهمیت ویژگی‌ها  

### 4) مشاهده تعاملی پیش‌بینی‌های 2025

```bash
python src/predict_for_date.py
```

**خروجی:**  
برنامه به‌صورت تعاملی تاریخ را می‌پرسد و برای آن تاریخ:

- دمای واقعی و پیش‌بینی‌شده  
- خطای مطلق  
- مقدار ویژگی‌های ورودی مهم (مثل `tavg`، `tavg_lag_1`، `tavg_roll_30_mean` و ...) را نمایش می‌دهد.

### 5) تحلیل روزهای با خطای بزرگ

```bash
python src/analyze_errors.py
```

**خروجی:**

- سه روز با بیشترین خطای مطلق در سال 2025  
- برای هر یک از این روزها:
  - تاریخ هدف و تاریخ ورودی  
  - دمای پیش‌بینی‌شده و واقعی  
  - خطای مطلق و جهت خطا (بیش‌برآورد / کم‌برآورد)  
  - شرایط هواشناسی سه روز قبل و روز هدف  
- فایل خروجی: `outputs/high_error_days_model_comparison.csv` شامل:
  - تاریخ روزهای با خطای بزرگ  
  - دمای واقعی و پیش‌بینی‌شده برای همهٔ مدل‌ها  
  - خطای مطلق برای هر مدل  
  - بهترین مدل برای هر روز (کمترین خطای مطلق)

## تحلیل خطاها (خلاصه)

خطاهای بزرگ معمولاً در زمان‌های کاهش ناگهانی دما یا تغییر سریع شرایط جوی رخ داده‌اند. مدل در این شرایط تمایل به کم‌برآورد کردن شدت تغییرات دارد.

## محدودیت‌ها

- مدل فقط برای تهران و با داده‌های روزانه آموزش دیده است.  
- پیش‌بینی بر مبنای مشاهدات روز جاری است و جایگزین مدل‌های عملیاتی پیش‌بینی عددی هواشناسی نیست.  
- در شرایط تغییرات ناگهانی جوی، خطا افزایش می‌یابد.

## منبع داده

Open-Meteo. Historical Weather API.  
داده‌ها از طریق API آرشیو Open-Meteo و با استفاده از اسکریپت `src/download_data.py` دریافت می‌شوند.

- وب‌سایت: https://open-meteo.com/  
- مستندات API: https://open-meteo.com/en/docs/historical-weather-api

[نسخه فارسی](README.fa.md) | [English version](README.md)