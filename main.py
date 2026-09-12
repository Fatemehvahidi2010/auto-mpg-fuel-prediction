import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import joblib


# خواندن دیتاست
df = pd.read_csv("auto-mpg.data", sep=r"\s+", header=None)


# نام ستون‌ها
df.columns = [
    "mpg",
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model_year",
    "origin",
    "car_name"
]


# نمایش چند سطر اول
st.dataframe(df.head())


# اطلاعات دیتاست
st.write("اطلاعات دیتاست:")
st.write()


# اطلاعات آماری دیتاست
st.write("اطلاعات آماری:")
st.dataframe(df.describe())


# تبدیل horsepower به عدد
df["horsepower"] = pd.to_numeric(
    df["horsepower"],
    errors="coerce"
)


# حذف داده‌های ناقص
df = df.dropna()


# تبدیل origin به چند ستون
df = pd.get_dummies(
    df,
    columns=["origin"],
    dtype=int
)


# انتخاب ویژگی‌ها
Features_cols = [
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model_year",
    "origin_1",
    "origin_2",
    "origin_3"
]


# تعیین X و Y
x = df[Features_cols]

y = df["mpg"]


# تقسیم داده‌ها به آموزش و تست
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)


# ساخت مدل درخت تصمیم
model = DecisionTreeRegressor(
    max_depth=5,
    random_state=42
)


# آموزش مدل
model.fit(x_train, y_train)


# پیش‌بینی
mpg_pred = model.predict(x_test)


# نمایش مقدار واقعی و پیش‌بینی شده
st.dataframe({
    "مصرف واقعی": y_test,
    "مصرف پیش بینی شده": mpg_pred
})


# عنوان برنامه
st.title("پیش بینی مصرف سوخت خودرو")


# توضیحات ویژگی‌ها
st.write("""
*توضیحات ویژگی‌ها:*

- cylinders: تعداد سیلندرهای خودرو
- displacement: حجم موتور
- horsepower: قدرت موتور
- weight: وزن خودرو
- acceleration: شتاب خودرو
- model_year: سال تولید خودرو
- origin: کشور سازنده خودرو

*هدف (Target):*

mpg = مصرف سوخت خودرو
""")


# محاسبه خطای MAE
error_men = mean_absolute_error(
    y_test,
    mpg_pred
)


st.write("خطای MAE:")
st.write(error_men)


# نمودار اول: هیستوگرام MPG
fig1, ax1 = plt.subplots()

ax1.hist(df["mpg"])

ax1.set_title("توزیع مصرف سوخت خودروها")
ax1.set_xlabel("MPG")
ax1.set_ylabel("تعداد خودروها")

st.pyplot(fig1)


# نمودار دوم: وزن خودرو و MPG
fig2, ax2 = plt.subplots()

ax2.scatter(
    df["weight"],
    df["mpg"]
)

ax2.set_title("رابطه وزن خودرو و مصرف سوخت")
ax2.set_xlabel("Weight")
ax2.set_ylabel("MPG")

st.pyplot(fig2)


# نمودار سوم: MPG بر اساس تعداد سیلندر
fig3, ax3 = plt.subplots()

df.boxplot(
    column="mpg",
    by="cylinders",
    ax=ax3
)

ax3.set_title("مصرف سوخت بر اساس تعداد سیلندر")
ax3.set_xlabel("Cylinders")
ax3.set_ylabel("MPG")

plt.suptitle("")

st.pyplot(fig3)


# رسم درخت تصمیم
fig4, ax4 = plt.subplots(
    figsize=(15, 8)
)

plot_tree(
    model,
    feature_names=Features_cols,
    filled=True,
    ax=ax4
)

st.pyplot(fig4)


# اهمیت ویژگی‌ها
fig5, ax5 = plt.subplots()

ax5.bar(
    Features_cols,
    model.feature_importances_
)

ax5.set_title("اهمیت ویژگی‌ها")
ax5.set_xlabel("ویژگی‌ها")
ax5.set_ylabel("اهمیت")

plt.xticks(rotation=45)

st.pyplot(fig5)

important_feature = Features_cols[
    model.feature_importances_.argmax()
]

st.write("مهم‌ترین ویژگی:")
st.write(important_feature)


# ذخیره مدل
joblib.dump(
    model,
    "fuel_model.joblib"
)


st.success("مدل با موفقیت ذخیره شد.")