import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Set the title for the Streamlit app
st.title("Valentines' Consumer Data")
st.write("Hey! Interested in seeing how people spend on Valentine's this is the right place!")
#valentines = Image.open('valantine_App.png')
st.image("valantine_App.jpg")
df = pd.read_csv("historical_spending.csv")

# Display a header for the Visualization section
st.markdown("Visualization: Amazing Right!!")


## Description of Dataset

num = st.number_input('No of Rows',5,20)
st.dataframe(df.head(num))

### Description of the dataset

st.dataframe(df.describe())

### Missing value

dfnull = df.isnull().sum()/len(df)*100
totalmiss = dfnull.sum().round(2)
st.write("Percentage of missing value in my dataset",totalmiss)

if st.button("Generate Report"):
  import streamlit as st
  import streamlit.components.v1 as components

  # Title for your app
  st.title('Sweetviz Report in Streamlit')
  # Display the Sweetviz report
  report_path = 'customer_report.html'
  HtmlFile = open(report_path, 'r', encoding='utf-8')
  source_code = HtmlFile.read()
  components.html(source_code, height=800,width=800)

list_variables=df.columns

filters = st.multiselect("Select two variables for Visualization:", list_variables, ["Flowers", "Year"])

tab1, tab2, tab3 = st.tabs(["Line Chart", "Bar Chart", "Filtered Chart"])
tab1.subheader("Line chart")
tab2.subheader("Bar chart")

# Display a line chart for the selected variables
tab1.line_chart(data=df, x=filters[0], y=filters[1], width=0, height=0, use_container_width=True)
#tab1.line_chart(data=df, x="Year", y="PerPerson")

# Display a bar chart for the selected variables
#tab2.bar_chart(data=df, x="Year", y="Flowers")
tab2.bar_chart(data=df, x=filters[0], y=filters[1], width=0, height=0, use_container_width=True)

# Filtering the dataframe based on the slider values
tab3.subheader('Flowers VS Years')
tab3.write("Please choose the range on the sidebar for specific years!")
Lowerbound_Year, Upperbound_Year = st.sidebar.slider("Select the Year's range for Visualization", min_value=int(df['Year'].min()), max_value=int(df['Year'].max()), value=(int(df['Year'].min()), int(df['Year'].max())))
Flowers_min, Flowers_max = st.sidebar.slider('Select the range value for Flowers', min_value=int(df['Flowers'].min()), max_value=int(df['Flowers'].max()), value=(int(df['Flowers'].min()), int(df['Flowers'].max())))
filtered_df = df[(df['Year'] >= Lowerbound_Year) & (df['Year'] <= Upperbound_Year) & (df['Flowers'] >= Flowers_min) & (df['Flowers'] <= Flowers_max)]

tab3.bar_chart(data=filtered_df, x="Year", y="Flowers")

