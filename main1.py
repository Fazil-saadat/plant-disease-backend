from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image

IMG_SIZE = 160
MODEL_PATH = "model/model.keras"
CLASSES_JSON = "metadata/classes.json"
UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load classes and treatments
with open(CLASSES_JSON, "r") as f:
    classes_dict = json.load(f)

app = FastAPI(title="Plant Disease Classifier")

# Allow mobile app or frontend to access the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directories
app.mount("/css", StaticFiles(directory="static/css"), name="css")
app.mount("/js", StaticFiles(directory="static/js"), name="js")
app.mount("/images", StaticFiles(directory="static/images"), name="images")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Multi-language disease database
# disease_data.py
DISEASE_DATABASE = {
    "Apple___Apple_scab": {
        "en": {
            "disease_name": "Apple Scab",
            "description": "A fungal disease caused by Venturia inaequalis that affects apple trees, causing dark, scaly lesions on leaves, fruits, and twigs.",
            "symptoms": "• Olive-green to black spots on leaves\n• Velvety, dark lesions on fruits\n• Yellowing and premature leaf drop\n• Cracked and deformed fruits\n• Twig lesions and cankers",
            "treatment": "• Apply fungicides like captan, myclobutanil, or sulfur\n• Remove and destroy infected leaves and fruits\n• Prune trees for better air circulation\n• Use resistant apple varieties like Liberty or Freedom\n• Apply dormant sprays in early spring",
            "prevention": "• Plant resistant apple varieties\n• Ensure proper tree spacing (15-20 feet apart)\n• Clean up fallen leaves in autumn\n• Avoid overhead watering\n• Apply preventative fungicides before infection"
        },
        "fa": {
            "disease_name": "زنگار سیب",
            "description": "یک بیماری قارچی ناشی از Venturia inaequalis که درختان سیب را تحت تاثیر قرار می‌دهد و باعث ایجاد زخم‌های تیره و پوسته پوسته بر روی برگ‌ها، میوه‌ها و شاخه‌ها می‌شود.",
            "symptoms": "• لکه‌های زیتونی تا سیاه روی برگ‌ها\n• زخم‌های مخملی و تیره روی میوه‌ها\n• زردی و ریزش زودرس برگ‌ها\n• میوه‌های ترک خورده و بدشکل\n• زخم و شانکر روی شاخه‌ها",
            "treatment": "• استفاده از قارچ‌کش‌هایی مانند کاپتان، میکلوبوتانیل یا گوگرد\n• حذف و نابودی برگ‌ها و میوه‌های آلوده\n• هرس درختان برای گردش هوای بهتر\n• استفاده از انواع مقاوم سیب مانند لیبرتی یا فریدم\n• استفاده از اسپری‌های خواب در اوایل بهار",
            "prevention": "• کاشت انواع مقاوم سیب\n• اطمینان از فاصله مناسب بین درختان (۱۵-۲۰ فوت)\n• تمیز کردن برگ‌های ریخته در پاییز\n• جلوگیری از آبیاری از بالا\n• استفاده از قارچ‌کش‌های پیشگیرانه قبل از عفونت"
        },
        "ps": {
            "disease_name": "د سیب زنگار",
            "description": "د Venturia inaequalis په واسطه یوه فنجي ناروغي ده چې د سیب ونی اغیزه کوي او د پاڼو، میوو او څاخو په سر تیاره او پوړ پوړ زخمونه رامنځته کوي.",
            "symptoms": "• د زیتون څخه تورو ته د پاڼو په سر داغونه\n• د میوو په سر مخملي، تیاره زخمونه\n• د پاڼو ژیړوالی او مخکینۍ لوېدل\n• ماتې شوې او بې شکلې میوې\n• د څاخو په سر زخمونه او شانکرونه",
            "treatment": "• د کاپتان، میکلوبوتانیل یا ګوګړ په څیر فنجي وژونکي کارول\n• ناروغه پاڼې او میوې لرې کول او ویجاړول\n• د هوا د غوره تبادلې لپاره ونی پرې کول\n• د سیب د مقاومو ډولونو کارول لکه لیبرتی یا فریدم\n• د پسرلي په لومړیو کې د خوب اسپري کارول",
            "prevention": "• د سیب مقاوم ډولونه کرل\n• د ونو ترمنځ د مناسب فاصلې ډاډمنول (۱۵-۲۰ فټه)\n• په مني کې د تلو پاڼو پاکول\n• د پاسه اوبو کولو څخه مخنیوی\n• د ناروغۍ مخنیوي لپاره فنجي وژونکي کارول"
        }
    },

    "Apple___Black_rot": {
        "en": {
            "disease_name": "Apple Black Rot",
            "description": "A fungal disease caused by Botryosphaeria obtusa that affects apples, causing fruit rot, leaf spots, and cankers on branches.",
            "symptoms": "• Frogeye leaf spots with purple margins\n• Black, rotting fruits with concentric rings\n• Red-brown cankers on branches\n• Premature fruit drop\n• Mummified fruits hanging on tree",
            "treatment": "• Prune and destroy infected branches\n• Apply fungicide sprays during growing season\n• Remove mummified fruits from tree\n• Improve air circulation through pruning\n• Use copper-based fungicides",
            "prevention": "• Practice good orchard sanitation\n• Remove infected plant material promptly\n• Avoid wounding fruits during handling\n• Use proper pruning techniques\n• Maintain tree vigor with proper nutrition"
        },
        "fa": {
            "disease_name": "پوسیدگی سیاه سیب",
            "description": "یک بیماری قارچی ناشی از Botryosphaeria obtusa که سیب را تحت تاثیر قرار می‌دهد و باعث پوسیدگی میوه، لکه‌های برگ و شانکر روی شاخه‌ها می‌شود.",
            "symptoms": "• لکه‌های چشم قورباغه‌ای روی برگ با حاشیه بنفش\n• میوه‌های سیاه و پوسیده با حلقه‌های متحدالمرکز\n• شانکرهای قهوه‌ای مایل به قرمز روی شاخه‌ها\n• ریزش زودرس میوه\n• میوه‌های مومیایی شده آویزان روی درخت",
            "treatment": "• هرس و نابودی شاخه‌های آلوده\n• استفاده از اسپری‌های قارچ‌کش در طول فصل رشد\n• حذف میوه‌های مومیایی شده از درخت\n• بهبود گردش هوا از طریق هرس\n• استفاده از قارچ‌کش‌های مبتنی بر مس",
            "prevention": "• رعایت بهداشت خوب باغ\n• حذف سریع مواد گیاهی آلوده\n• جلوگیری از زخمی شدن میوه‌ها در حین جابجایی\n• استفاده از تکنیک‌های هرس مناسب\n• حفظ قدرت درخت با تغذیه مناسب"
        },
        "ps": {
            "disease_name": "د سیب تور پوسیدگی",
            "description": "د Botryosphaeria obtusa په واسطه یوه فنجي ناروغي ده چې سیب اغیزه کوي او د میوو پوسیدگی، د پاڼو داغونه او د څانګو په سر شانکرونه رامنځته کوي.",
            "symptoms": "• د پاڼو په سر د چنجې ښکارۍ داغونه د ارغواني څنډو سره\n• تورې، پوسیدلې میوې د متمرکزو حلقو سره\n• د څانګو په سر سور-نسواري شانکرونه\n• د میوو مخکینۍ لوېدل\n• په ونه کې د مومیایی شویو میوو ځوړندېدل",
            "treatment": "• ناروغې څانګې پرې کول او ویجاړول\n• د ودی په موسم کې د فنجي وژونکو اسپري کارول\n• د ونی څخه مومیایی شوې میوې لرې کول\n• د پرې کولو له لارې د هوا تبادله ښه کول\n• د مس پر بنسټ فنجي وژونکي کارول",
            "prevention": "• د باغ ښه بهداشت تمرین کول\n• په چټکۍ سره ناروغه نباتي مواد لرې کول\n• د میوو د لاسرسي په وخت کې د میوو زخمي کولو څخه مخنیوی\n• د مناسب پرې کولو تخنیکونو کارول\n• د مناسب تغذیې سره د ونی قوت ساتل"
        }
    },

    "Apple___Cedar_apple_rust": {
        "en": {
            "disease_name": "Cedar Apple Rust",
            "description": "A fungal disease caused by Gymnosporangium juniperi-virginianae that requires both apple trees and juniper plants to complete its life cycle.",
            "symptoms": "• Yellow-orange spots on upper leaf surfaces\n• Tube-like structures on lower leaf surfaces\n• Cedar galls on juniper trees\n• Premature leaf drop\n• Reduced fruit quality and yield",
            "treatment": "• Remove nearby juniper hosts within 2 miles\n• Apply protective fungicides in early spring\n• Use resistant apple varieties\n• Prune infected branches\n• Apply fungicides at pink bud stage",
            "prevention": "• Plant resistant apple varieties\n• Eliminate juniper hosts in vicinity\n• Apply preventative fungicides before symptoms appear\n• Monitor trees regularly for early detection\n• Improve air circulation around trees"
        },
        "fa": {
            "disease_name": "زنگار سدر و سیب",
            "description": "یک بیماری قارچی ناشی از Gymnosporangium juniperi-virginianae که برای تکمیل چرخه زندگی خود به هر دو درخت سیب و گیاهان سدر نیاز دارد.",
            "symptoms": "• لکه‌های زرد-نارنجی روی سطوح بالایی برگ\n• ساختارهای لوله‌ای روی سطوح زیرین برگ\n• گال‌های سدر روی درختان سدر\n• ریزش زودرس برگ\n• کاهش کیفیت و عملکرد میوه",
            "treatment": "• حذف میزبان‌های سدر مجاور در فاصله ۲ مایلی\n• استفاده از قارچ‌کش‌های محافظ در اوایل بهار\n• استفاده از انواع مقاوم سیب\n• هرس شاخه‌های آلوده\n• استفاده از قارچ‌کش در مرحله غنچه صورتی",
            "prevention": "• کاشت انواع مقاوم سیب\n• حذف میزبان‌های سدر در مجاورت\n• استفاده از قارچ‌کش‌های پیشگیرانه قبل از ظهور علائم\n• نظارت منظم بر درختان برای تشخیص زودرس\n• بهبود گردش هوا در اطراف درختان"
        },
        "ps": {
            "disease_name": "د سدر او سیب زنگار",
            "description": "د Gymnosporangium juniperi-virginianae په واسطه یوه فنجي ناروغي ده چې د خپل د ژوند دوره بشپړولو لپاره د سیب د ونی او د سدر نباتاتو ته اړتیا لري.",
            "symptoms": "• د پاڼو د پورتنۍ سطحو په سر ژیړ-نارنجي داغونه\n• د پاڼو د لاندینۍ سطحو په سر د ټیوب په څیر ساختمانونه\n• د سدر په ونسو کې د سدر ګالونه\n• د پاڼو مخکینۍ لوېدل\n• د میوو د کیفیت او حاصل کمښت",
            "treatment": "• د ۲ مایلونو په فاصله کې نږدې سدر میزبانان لرې کول\n• د پسرلي په لومړیو کې د ساتونکو فنجي وژونکو کارول\n• د سیب مقاوم ډولونه کارول\n• ناروغې څانګې پرې کول\n• د ګلابي غوټۍ په مرحله کې فنجي وژونکي کارول",
            "prevention": "• د سیب مقاوم ډولونه کرل\n• په شاوخوا کې د سدر میزبانان له منځه وړل\n• د نښو د څرګندیدو څخه مخکې د مخنیوي فنجي وژونکي کارول\n• د لومړنۍ تشخیص لپاره په منظم ډول ونی څارل\n• د ونی شاوخوا کې د هوا تبادله ښه کول"
        }
    },

    "Apple___Healthy": {
        "en": {
            "disease_name": "Healthy Apple",
            "description": "Your apple tree appears to be in excellent health with no signs of common diseases. It shows vigorous growth and proper development.",
            "symptoms": "• Vibrant green leaves with no discoloration\n• Normal fruit development and size\n• Strong, flexible branches\n• No visible spots, lesions, or abnormalities\n• Consistent growth pattern",
            "treatment": "• No chemical treatment required\n• Continue regular watering schedule\n• Maintain proper fertilization\n• Monitor for early signs of pests\n• Practice seasonal pruning",
            "prevention": "• Continue current maintenance practices\n• Regular inspection for pests and diseases\n• Proper spacing between trees\n• Balanced nutrition and soil management\n• Seasonal care and protection"
        },
        "fa": {
            "disease_name": "سیب سالم",
            "description": "درخت سیب شما در سلامت عالی به نظر می‌رسد و هیچ نشانه‌ای از بیماری‌های شایع نشان نمی‌دهد. رشد قوی و توسعه مناسب را نشان می‌دهد.",
            "symptoms": "• برگ‌های سبز پرجنب و جوش بدون تغییر رنگ\n• رشد و اندازه طبیعی میوه\n• شاخه‌های قوی و انعطاف‌پذیر\n• بدون لکه، زخم یا ناهنجاری قابل مشاهده\n• الگوی رشد یکنواخت",
            "treatment": "• نیاز به درمان شیمیایی ندارد\n• ادامه برنامه آبیاری منظم\n• حفظ کوددهی مناسب\n• نظارت بر علائم اولیه آفات\n• انجام هرس فصلی",
            "prevention": "• ادامه روش‌های نگهداری فعلی\n• بازرسی منظم برای آفات و بیماری‌ها\n• فاصله مناسب بین درختان\n• تغذیه متعادل و مدیریت خاک\n• مراقبت و محافظت فصلی"
        },
        "ps": {
            "disease_name": "تندرسته سیب",
            "description": "ستاسو د سیب ونه په عالي توګه تندرسته ښکاري او د عامو ناروغیو هېڅ نښه نه ښیي. قوي وده او مناسب پراختیا ښیي.",
            "symptoms": "• د رنګ بدلون پرته ژوندۍ شینې پاڼې\n• د میوو عادي وده او کچه\n• قوي، انعطاف منونکې څانګې\n• هېڅ لیدونکی داغ، زخم یا غیرعادي حالتونه نه\n• د ودې یو ډول نمونه",
            "treatment": "• د کیمیاوي درملنې اړتیا نشته\n• د اوبو کولو منظم برنامه دوام ورکړئ\n• مناسب سره ورکول ساتل\n• د زیان رسوونکو د لومړنیو نښو لپاره څارنه\n• د موسمي پرې کولو تمرین کول",
            "prevention": "• اوسنۍ ساتنې طریقې دوام ورکړئ\n• د زیان رسوونکو او ناروغیو لپاره منظم معاینه\n• د ونو ترمنځ مناسب فاصله\n• متوازنه تغذیه او خاورې مدیریت\n• د موسمي پالنې او ساتنې"
        }
    },

    "Cherry___Powdery_mildew": {
        "en": {
            "disease_name": "Cherry Powdery Mildew",
            "description": "A fungal disease caused by Podosphaera clandestina that creates white powdery growth on cherry leaves, shoots, and fruits.",
            "symptoms": "• White powdery coating on leaves and shoots\n• Curled and distorted leaves\n• Stunted shoot growth\n• Reduced fruit quality and size\n• Premature leaf drop in severe cases",
            "treatment": "• Apply sulfur or potassium bicarbonate sprays\n• Use horticultural oils like neem oil\n• Remove severely infected leaves\n• Improve air circulation through pruning\n• Apply fungicides at first sign of infection",
            "prevention": "• Plant in sunny, well-ventilated locations\n• Ensure good air circulation around trees\n• Avoid overhead watering\n• Use resistant cherry varieties\n• Maintain proper tree spacing"
        },
        "fa": {
            "disease_name": "کپک پودری گیلاس",
            "description": "یک بیماری قارچی ناشی از Podosphaera clandestina که رشد پودری سفید روی برگ‌ها، شاخه‌ها و میوه‌های گیلاس ایجاد می‌کند.",
            "symptoms": "• پوشش پودری سفید روی برگ‌ها و شاخه‌ها\n• برگ‌های پیچ خورده و تغییر شکل یافته\n• رشد متوقف شده شاخه‌ها\n• کاهش کیفیت و اندازه میوه\n• ریزش زودرس برگ در موارد شدید",
            "treatment": "• استفاده از اسپری‌های گوگرد یا بی‌کربنات پتاسیم\n• استفاده از روغن‌های باغبانی مانند روغن نیم\n• حذف برگ‌های شدیداً آلوده\n• بهبود گردش هوا از طریق هرس\n• استفاده از قارچ‌کش در اولین نشانه عفونت",
            "prevention": "• کاشت در مکان‌های آفتابی و دارای تهویه مناسب\n• اطمینان از گردش هوای خوب در اطراف درختان\n• جلوگیری از آبیاری از بالا\n• استفاده از انواع مقاوم گیلاس\n• حفظ فاصله مناسب بین درختان"
        },
        "ps": {
            "disease_name": "د چیری پوډری میلډیو",
            "description": "د Podosphaera clandestina په واسطه یوه فنجي ناروغي ده چې د چیری په پاڼو، څانګو او میوو کې سپین پوډري وده رامنځته کوي.",
            "symptoms": "• د پاڼو او څانګو په سر سپین پوډري پوښ\n• تاو شوې او بې شکلې شوې پاڼې\n• د څانګو وده کمه شوې\n• د میوو د کیفیت او کچې کمښت\n• په شدیدو مواردو کې د پاڼو مخکینۍ لوېدل",
            "treatment": "• د ګوګړ یا پوتاشیم بایکاربونیټ اسپري کارول\n• د باغباني تیلو کارول لکه نیم تیل\n• شدید ناروغه پاڼې لرې کول\n• د پرې کولو له لارې د هوا تبادله ښه کول\n• د ناروغۍ په لومړۍ نښه کې فنجي وژونکي کارول",
            "prevention": "• په لمر لرونکو، ښه هوا لرونکو ځایونو کې کرل\n• د ونی شاوخوا کې د هوا د ښې تبادلې ډاډمنول\n• د پاسه اوبو کولو څخه مخنیوی\n• د چیری مقاوم ډولونه کارول\n• د ونو ترمنځ مناسب فاصله ساتل"
        }
    },

    "Cherry___Healthy": {
        "en": {
            "disease_name": "Healthy Cherry",
            "description": "Your cherry tree is in perfect health with no signs of disease. It exhibits strong growth, proper foliage, and normal fruit development.",
            "symptoms": "• Lush green foliage with no discoloration\n• Normal flowering and fruit set\n• Strong, well-structured branches\n• No powdery coating or spots\n• Vigorous growth throughout season",
            "treatment": "• No treatment necessary\n• Continue regular maintenance\n• Monitor for pest activity\n• Maintain proper watering\n• Apply balanced fertilization",
            "prevention": "• Continue current care practices\n• Regular inspection for issues\n• Proper pruning techniques\n• Soil health management\n• Seasonal monitoring and care"
        },
        "fa": {
            "disease_name": "گیلاس سالم",
            "description": "درخت گیلاس شما در سلامت کامل است و هیچ نشانه‌ای از بیماری نشان نمی‌دهد. رشد قوی، شاخ و برگ مناسب و توسعه طبیعی میوه را نشان می‌دهد.",
            "symptoms": "• شاخ و برگ سبز انبوه بدون تغییر رنگ\n• گلدهی و تشکیل میوه طبیعی\n• شاخه‌های قوی و دارای ساختار خوب\n• بدون پوشش پودری یا لکه\n• رشد قوی در طول فصل",
            "treatment": "• نیاز به درمان ندارد\n• ادامه نگهداری منظم\n• نظارت بر فعالیت آفات\n• حفظ آبیاری مناسب\n• استفاده از کوددهی متعادل",
            "prevention": "• ادامه روش‌های مراقبت فعلی\n• بازرسی منظم برای مشکلات\n• تکنیک‌های هرس مناسب\n• مدیریت سلامت خاک\n• نظارت و مراقبت فصلی"
        },
        "ps": {
            "disease_name": "تندرسته چیری",
            "description": "ستاسو د چیری ونه په کامل ډول تندرسته ده او د ناروغۍ هېڅ نښه نه ښیي. قوي وده، مناسب پاڼې او د میوو عادي پراختیا ښیي.",
            "symptoms": "• د رنګ بدلون پرته ډبرې شینې پاڼې\n• عادي ګل کول او میوه کول\n• قوي، ښه ساختمان لرونکې څانګې\n• هېڅ پوډري پوښ یا داغونه نه\n• د موسم په اوږدو کې قوي وده",
            "treatment": "• د درملنې اړتیا نشته\n• منظم ساتنه دوام ورکړئ\n• د زیان رسوونکو د فعالیت لپاره څارنه\n• مناسب اوبه کول ساتل\n• متوازن سره ورکول پلي کول",
            "prevention": "• اوسنۍ پالنې طریقې دوام ورکړئ\n• د ستونزو لپاره منظم معاینه\n• د مناسب پرې کولو تخنیکونه\n• د خاورې روغتیا مدیریت\n• د موسمي څارنې او پالنې"
        }
    },

    "Grape___Black_rot": {
        "en": {
            "disease_name": "Grape Black Rot",
            "description": "A serious fungal disease caused by Guignardia bidwellii that affects grapes, causing fruit rot and leaf spots, potentially destroying entire crops.",
            "symptoms": "• Brown leaf spots with black margins\n• Black, mummified fruits\n• Red-brown lesions on shoots\n• Premature fruit drop\n• Complete crop loss in severe cases",
            "treatment": "• Apply fungicides like mancozeb or captan\n• Remove and destroy infected plant material\n• Prune for better air circulation\n• Use protective sprays during flowering\n• Apply fungicides at 7-10 day intervals",
            "prevention": "• Plant resistant grape varieties\n• Ensure good vineyard sanitation\n• Proper canopy management\n• Avoid overhead irrigation\n• Regular monitoring and early treatment"
        },
        "fa": {
            "disease_name": "پوسیدگی سیاه انگور",
            "description": "یک بیماری قارچی جدی ناشی از Guignardia bidwellii که انگور را تحت تاثیر قرار می‌دهد و باعث پوسیدگی میوه و لکه‌های برگ می‌شود و می‌تواند کل محصول را نابود کند.",
            "symptoms": "• لکه‌های قهوه‌ای برگ با حاشیه سیاه\n• میوه‌های سیاه و مومیایی شده\n• زخم‌های قهوه‌ای مایل به قرمز روی شاخه‌ها\n• ریزش زودرس میوه\n• از دست دادن کامل محصول در موارد شدید",
            "treatment": "• استفاده از قارچ‌کش‌هایی مانند مانکوزب یا کاپتان\n• حذف و نابودی مواد گیاهی آلوده\n• هرس برای گردش هوای بهتر\n• استفاده از اسپری‌های محافظ در طول گلدهی\n• استفاده از قارچ‌کش با فواصل ۷-۱۰ روزه",
            "prevention": "• کاشت انواع مقاوم انگور\n• اطمینان از بهداشت خوب تاکستان\n• مدیریت مناسب سایبان\n• جلوگیری از آبیاری از بالا\n• نظارت منظم و درمان زودرس"
        },
        "ps": {
            "disease_name": "د انګور تور پوسیدگی",
            "description": "د Guignardia bidwellii په واسطه یوه جدي فنجي ناروغي ده چې انګور اغیزه کوي او د میوو پوسیدگی او د پاڼو داغونه رامنځته کوي او ممکن ټول محصول ویجاړ کړي.",
            "symptoms": "• د پاڼو نسواري داغونه د تورو څنډو سره\n• تورې، مومیایی شوې میوې\n• د څانګو په سر سور-نسواري زخمونه\n• د میوو مخکینۍ لوېدل\n• په شدیدو مواردو کې د محصول بشپړ ضایع کیدل",
            "treatment": "• د مانکوزب یا کاپتان په څیر فنجي وژونکي کارول\n• ناروغه نباتي مواد لرې کول او ویجاړول\n• د هوا د غوره تبادلې لپاره پرې کول\n• د ګل کولو په وخت کې د ساتونکو اسپري کارول\n• په ۷-۱۰ ورځني وقفو کې فنجي وژونکي کارول",
            "prevention": "• د انګور مقاوم ډولونه کرل\n• د تاکستان د ښه بهداشت ډاډمنول\n• مناسب سایبان مدیریت\n• د پاسه اوبو کولو څخه مخنیوی\n• منظم څارنه او لومړنۍ درملنه"
        }
    },

    "Grape___Esca_(Black_Measles)": {
        "en": {
            "disease_name": "Grape Esca (Black Measles)",
            "description": "A complex fungal disease caused by multiple pathogens that affects grapevines, leading to wood decay and various foliar symptoms.",
            "symptoms": "• Tiger-stripe patterns on leaves\n• Wood decay and cankers\n• Reduced vine vigor\n• Fruit spots and rotting\n• Sudden vine collapse (apoplexy)",
            "treatment": "• Prune infected wood below symptoms\n• Protect pruning wounds with fungicides\n• Remove severely infected vines\n• Improve vineyard drainage\n• Use balanced fertilization",
            "prevention": "• Use certified disease-free planting material\n• Proper pruning wound protection\n• Avoid mechanical injuries to vines\n• Maintain vine balance and health\n• Regular vineyard monitoring"
        },
        "fa": {
            "disease_name": "اسکای انگور (سرخک سیاه)",
            "description": "یک بیماری قارچی پیچیده ناشی از چندین پاتوژن که تاک‌های انگور را تحت تاثیر قرار می‌دهد و منجر به پوسیدگی چوب و علائم مختلف برگ می‌شود.",
            "symptoms": "• الگوهای راه راه ببر روی برگ‌ها\n• پوسیدگی چوب و شانکر\n• کاهش قدرت تاک\n• لکه‌های میوه و پوسیدگی\n• ریزش ناگهانی تاک (آپوپلکسی)",
            "treatment": "• هرس چوب آلوده زیر علائم\n• محافظت از زخم‌های هرس با قارچ‌کش\n• حذف تاک‌های شدیداً آلوده\n• بهبود زهکشی تاکستان\n• استفاده از کوددهی متعادل",
            "prevention": "• استفاده از مواد کاشت عاری از بیماری گواهی شده\n• محافظت مناسب از زخم هرس\n• جلوگیری از آسیب‌های مکانیکی به تاک‌ها\n• حفظ تعادل و سلامت تاک\n• نظارت منظم تاکستان"
        },
        "ps": {
            "disease_name": "د انګور اسکا (تورې پښتورۍ)",
            "description": "د څو پاتوجینونو په واسطه یوه پیچلې فنجي ناروغي ده چې د انګور تاکونه اغیزه کوي او د لرګیو پوسیدگی او مختلفې پاڼیزې نښې رامنځته کوي.",
            "symptoms": "• د پاڼو په سر د پښانګې د راه راه نمونه\n• د لرګیو پوسیدگی او شانکرونه\n• د تاک قوت کمښت\n• د میوو داغونه او پوسیدگی\n• د تاک ناڅاپي سقوط (اپوپلکسي)",
            "treatment": "• د نښو لاندې ناروغه لرګی پرې کول\n• د پرې کولو زخمونه د فنجي وژونکو سره ساتل\n• شدید ناروغه تاکونه لرې کول\n• د تاکستان تخلیه ښه کول\n• متوازن سره ورکول کارول",
            "prevention": "• د تصدیق شویو ناروغۍ څخه خوشې د کرلو موادو کارول\n• د پرې کولو زخمونو مناسب ساتنه\n• د تاکونو مکانیکي زیانونه مخنیوی\n• د تاک توازن او روغتیا ساتل\n• د تاکستان منظم څارنه"
        }
    },

    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "en": {
            "disease_name": "Grape Leaf Blight",
            "description": "A fungal disease caused by Pseudocercospora vitis that affects grape leaves, causing spots and premature defoliation.",
            "symptoms": "• Angular brown spots on leaves\n• Yellow halos around spots\n• Premature leaf drop\n• Reduced fruit quality\n• Weakened vine growth",
            "treatment": "• Apply copper-based fungicides\n• Remove infected leaves\n• Improve air circulation\n• Use protective sprays\n• Maintain vine health",
            "prevention": "• Proper vineyard sanitation\n• Good air circulation\n• Avoid overhead watering\n• Regular monitoring\n• Balanced nutrition"
        },
        "fa": {
            "disease_name": "بلایت برگ انگور",
            "description": "یک بیماری قارچی ناشی از Pseudocercospora vitis که برگ‌های انگور را تحت تاثیر قرار می‌دهد و باعث ایجاد لکه و ریزش زودرس برگ می‌شود.",
            "symptoms": "• لکه‌های قهوه‌ای زاویه‌دار روی برگ‌ها\n• هاله‌های زرد اطراف لکه‌ها\n• ریزش زودرس برگ\n• کاهش کیفیت میوه\n• تضعیف رشد تاک",
            "treatment": "• استفاده از قارچ‌کش‌های مبتنی بر مس\n• حذف برگ‌های آلوده\n• بهبود گردش هوا\n• استفاده از اسپری‌های محافظ\n• حفظ سلامت تاک",
            "prevention": "• بهداشت مناسب تاکستان\n• گردش هوای خوب\n• جلوگیری از آبیاری از بالا\n• نظارت منظم\n• تغذیه متعادل"
        },
        "ps": {
            "disease_name": "د انګور د پاڼې بلایت",
            "description": "د Pseudocercospora vitis په واسطه یوه فنجي ناروغي ده چې د انګور پاڼې اغیزه کوي او داغونه او د پاڼو مخکینۍ لوېدل رامنځته کوي.",
            "symptoms": "• د پاڼو په سر زاویه لرونکې نسواري داغونه\n• د داغونو شاوخوا ژیړ هاله\n• د پاڼو مخکینۍ لوېدل\n• د میوو د کیفیت کمښت\n• د تاک ودې کمزوري کیدل",
            "treatment": "• د مس پر بنسټ فنجي وژونکي کارول\n• ناروغه پاڼې لرې کول\n• د هوا تبادله ښه کول\n• د ساتونکو اسپري کارول\n• د تاک روغتیا ساتل",
            "prevention": "• د تاکستان مناسب بهداشت\n• د هوا ښه تبادله\n• د پاسه اوبو کولو څخه مخنیوی\n• منظم څارنه\n• متوازنه تغذیه"
        }
    },

  "Grape___Healthy": {
    "en": {
        "disease_name": "Healthy Grape",
        "description": "Your grape vine appears to be healthy and free from common diseases.",
        "symptoms": "• Green, vibrant leaves\n• Normal fruit cluster development\n• No visible spots or lesions\n• Strong vine growth\n• Healthy tendrils",
        "treatment": "• No treatment needed\n• Continue regular care\n• Monitor for early signs of disease\n• Maintain proper nutrition\n• Ensure adequate sunlight",
        "prevention": "• Continue good cultural practices\n• Regular watering and fertilization\n• Monitor for pests and diseases\n• Proper pruning and trellising\n• Good air circulation"
    },
    "fa": {
        "disease_name": "انگور سالم",
        "description": "درخت انگور شما سالم به نظر می‌رسد و از بیماری‌های شایع آزاد است.",
        "symptoms": "• برگ‌های سبز و پرجنب و جوش\n• رشد طبیعی خوشه‌های میوه\n• بدون لکه یا زخم قابل مشاهده\n• رشد قوی درخت\n• پیچک‌های سالم",
        "treatment": "• نیاز به درمان ندارد\n• مراقبت منظم را ادامه دهید\n• نظارت بر علائم اولیه بیماری\n• حفظ تغذیه مناسب\n• اطمینان از نور کافی خورشید",
        "prevention": "• ادامه روش‌های فرهنگی خوب\n• آبیاری و کوددهی منظم\n• نظارت بر آفات و بیماری‌ها\n• هرس و داربست‌بندی مناسب\n• گردش هوای خوب"
    },
    "ps": {
        "disease_name": "تندرسته انگور",
        "description": "ستاسو د انگور ونه تندرسته ښکاري او د عامو ناروغیو څخه خوشې ده.",
        "symptoms": "• شینې، ژوندۍ پاڼې\n• د میوو د ګڼو عادي وده\n• هېڅ لیدونکی داغ یا زخم نه\n• د ونې قوي وده\n• تندرسته پیچکونه",
        "treatment": "• د درملنې اړتیا نشته\n• منظم پالنه دوام ورکړئ\n• د ناروغۍ د لومړنیو نښو لپاره څارنه\n• مناسب تغذیه ساتل\n• د لمر د کافي رڼا ډاډمنول",
        "prevention": "• د ښو کلتوري طریقو دوام\n• منظم اوبه کول او سره ورکول\n• د زیان رسوونکو او ناروغیو لپاره څارنه\n• مناسب پرې کول او داربست کول\n• د هوا ښه جریان"
    }
}
}

def get_localized_disease_info(class_name: str, language: str = "en"):
    """
    Get localized disease information for the given class name and language
    """
    # Default response if disease not found
    default_response = {
        "disease_name": class_name,
        "description": "Disease information not available",
        "symptoms": "No symptoms information available",
        "treatment": "No treatment information available", 
        "prevention": "No prevention information available"
    }
    
    # Return disease info if found, otherwise default
    if class_name in DISEASE_DATABASE:
        disease_info = DISEASE_DATABASE[class_name]
        # Return requested language or English as fallback
        return disease_info.get(language, disease_info['en'])
    else:
        return default_response

# Utility
def preprocess_image(file_path):
    img = Image.open(file_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Routes
@app.get("/", response_class=HTMLResponse)
async def home():
    with open(os.path.join(os.path.dirname(__file__), "templates/base.html"), "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    language: str = Form("en")  # Language parameter with default 'en'
):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        img_array = preprocess_image(file_path)
        preds = model.predict(img_array)
        pred_idx = int(np.argmax(preds, axis=1)[0])
        confidence = float(np.max(preds)) * 100

        class_info = classes_dict.get(str(pred_idx), {})
        pred_class = class_info.get("class_name", "Unknown")
        
        # Get localized disease information
        localized_info = get_localized_disease_info(pred_class, language)
        
        print(f"🚀 Prediction: {pred_class}, Language: {language}, Confidence: {confidence:.2f}%")

        return JSONResponse(
            content={
                "prediction": pred_class,
                "disease_name": localized_info["disease_name"],
                "description": localized_info["description"],
                "symptoms": localized_info["symptoms"],
                "treatment": localized_info["treatment"],
                "prevention": localized_info["prevention"],
                "confidence": round(confidence, 2),
                "uploaded_image": f"/uploads/{file.filename}",
                "language": language
            }
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# New endpoint to get supported languages
@app.get("/languages")
async def get_supported_languages():
    """Return list of supported languages"""
    return {
        "supported_languages": [
            {"code": "en", "name": "English"},
            {"code": "fa", "name": "فارسی"}, 
            {"code": "ps", "name": "پښتو"}
        ]
    }

# New endpoint to get all diseases for a specific language
@app.get("/diseases/{language}")
async def get_all_diseases(language: str = "en"):
    """Get all disease information for a specific language"""
    result = {}
    for disease_key in DISEASE_DATABASE:
        result[disease_key] = get_localized_disease_info(disease_key, language)
    return result
# Run the app