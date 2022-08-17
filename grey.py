import pandas as pd 
import streamlit as st
import pandas as pd
import io
import numpy as np
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
import zipfile
import base64
import os
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import neattext as nt
from autocorrect import Speller
import spacy
en_core = spacy.load('en_core_web_sm')
spell = Speller(lang='en')
import plotly.express as px
from PIL import Image
from textblob import TextBlob, Word
from textblob import TextBlob
from streamlit_option_menu import option_menu
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO
from wordcloud import WordCloud 
from matplotlib import colors         
from nltk.corpus import stopwords           
color_list=['#4d4d4d','#ffcc00','#000000',]            
colormap=colors.ListedColormap(color_list)            
import matplotlib.pyplot as plt            
            
            




def main_page():
    st.markdown("# Main page 🎈")
    st.sidebar.markdown("# Main page 🎈")
    
def page2():
    st.markdown("# Filter 🎈")
    st.sidebar.markdown("# Filter Text 🎈")


    multiple_files = st.file_uploader(" ",type="csv", accept_multiple_files=True)
    for file in multiple_files:
        df = pd.read_csv(file)
        file.seek(0)
    
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_pagination(paginationAutoPageSize=True) #Add pagination
        gb.configure_side_bar() #Add a sidebar
        gb.configure_selection('multiple') #Enable multi-row selection
        gridOptions = gb.build()

        grid_response = AgGrid(
            df,
            gridOptions=gridOptions,
            data_return_mode='AS_INPUT', 
            update_mode='MODEL_CHANGED', 
            fit_columns_on_grid_load=False,
            theme='blue', #Add theme color to the table
            enable_enterprise_modules=True,
            height=600, 
            width='100%',
            reload_data=True
        )



        data = grid_response['data']
        selected = grid_response['selected_rows'] 
        df = pd.DataFrame(selected) #Pass the selected rows to a new dataframe df
        
        ###################################################################

def page3():
    st.markdown("# Merge  ❄️")
    st.sidebar.markdown("# Merge Files ❄️")
    
    # Excel file merge function
    def excel_file_merge(zip_file_name):
        df = pd.DataFrame()
        archive = zipfile.ZipFile(zip_file_name, 'r')
        with zipfile.ZipFile(zip_file_name, "r") as f:
            for file in f.namelist():
              xlfile = archive.open(file)
              if file.endswith('.xlsx'):
                # Add a note indicating the file name that this dataframe originates from
                df_xl = pd.read_excel(xlfile, engine='openpyxl')
                df_xl['Note'] = file
                # Appends content of each Excel file iteratively
                df = df.append(df_xl, ignore_index=True)
        return df

    # Upload CSV data
    with st.sidebar.header('1. Upload your ZIP file'):
        uploaded_file = st.sidebar.file_uploader("Excel-containing ZIP file", type=["zip"])
    

    # File download
    def filedownload(df):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
        href = f'<a href="data:file/csv;base64,{b64}" download="merged_file.csv">Download Merged File as CSV</a>'
        return href


    # Main panel
    if st.sidebar.button('Submit'):
        #@st.cache
        df = excel_file_merge(uploaded_file)
        st.header('**Merged data**')
        st.write(df)
        st.markdown(filedownload(df), unsafe_allow_html=True)
    else:
        st.info('Awaiting for ZIP file to be uploaded.')
        
        ########################################################################

def page4():
    st.markdown("# Profiling Report 🎉")
    st.sidebar.markdown("# Profiling Report 🎉")

    # Upload CSV data
    with st.sidebar.header('Upload your CSV data'):
        uploaded_file = st.sidebar.file_uploader(" ", type=["csv"])
        st.sidebar.markdown("""

    """)

    # Pandas Profiling Report
    if uploaded_file is not None:
        @st.cache
        def load_csv():
            csv = pd.read_csv(uploaded_file)
            return csv
        df = load_csv()
        pr = ProfileReport(df, explorative=True)
        st.header('**Input DataFrame**')
        st.write(df)
        st.write('---')
        st.header('**Pandas Profiling Report**')
        st_profile_report(pr)
    else:
        st.info('Awaiting for CSV file to be uploaded.')
            # Example data
        @st.cache
        def load_data():
            df = load_data()
            pr = ProfileReport(df, explorative=True)
            st.header('**Input DataFrame**')
            st.write(df)
            st.write('---')
            st.header('**Pandas Profiling Report**')
            st_profile_report(pr)
            
            ##################################################################
    
def page5():
    st.markdown("# Renaming Columns 🎉")
    st.sidebar.markdown("# Renaming Columns 🎉")
    
    text_file = st.file_uploader("Upload CSV File", type=['csv'])
    if text_file is not None:
        df= pd.read_csv(text_file)
        st.write(df)

        with st.form(key="form"):
            col_to_change = st.selectbox("Column to change", df.columns)
            new_col_name = st.text_input("New name", value="")
            submit_button = st.form_submit_button(label='Rename')

        if submit_button:
            df = df.rename(columns={col_to_change: new_col_name})

            st.write(df)
            clean_text = df['Text']
               
            st.download_button(label='Download Dataset',data=clean_text.to_csv(),file_name='renamedfile.csv',mime='text/csv')

        #####################################################################
        
def page6():
    st.markdown("# Text Cleaner 3 🎉")
    st.sidebar.markdown("# Clean text 🎉")
    
    def main():
            
        text_file = st.file_uploader("Upload CSV File",type=['csv']) 
 
        if text_file is not None:
            cleanerdf= pd.read_csv(text_file)
            cleanerdf['Cleaned_Text'] = cleanerdf['Text'].apply(str.lower)
            cleanerdf["Cleaned_Text"] = cleanerdf['Cleaned_Text'].apply(lambda x: " ".join([y.lemma_ for y in en_core(x)]))
        
            cdf1 = cleanerdf.copy()
        
            cdf1['Cleaned_Text'] = [' '.join([spell(i) for i in x.split()]) for x in cdf1['Cleaned_Text']]
            cdf1['Cleaned_Text'] = cdf1['Cleaned_Text'].apply(nt.remove_punctuations)
            cdf1['Cleaned_Text'] = cdf1['Cleaned_Text'].apply(nt.remove_numbers)
            cdf1['Cleaned_Text'] = cdf1['Cleaned_Text'].apply(nt.remove_special_characters)
            cdf1['Cleaned_Text'] = cdf1['Cleaned_Text'].apply(nt.remove_dates)
            cdf1['Cleaned_Text'] = cdf1['Cleaned_Text'].apply(nt.remove_emojis)
            cdf1['Cleaned_Text'] = cdf1['Cleaned_Text'].apply(nt.remove_stopwords)
      
            st.write(cdf1)
            clean_text = cdf1['Cleaned_Text']

    
            st.download_button(label='Download Cleaned Text',data=clean_text.to_csv(),file_name='cleantext.csv',mime='text/csv')
            
        
    if __name__ == '__main__':
        main()
        
                #####################################################################

        
def page7():
    st.markdown("# Word Frequency 🎉")
    st.sidebar.markdown(" ")
    
            
    text_file = st.file_uploader("Upload CSV File",type=['csv'])

    col1, col2 = st.columns(2)

    with col1:
        if text_file is not None:
            st.markdown('''
            # **Cleaned Text**
            ---
            ''')        
            wdf= pd.read_csv(text_file)
            col1.write(wdf)
      
            wdf["text"] = wdf['Cleaned_Text']
            new_df = wdf.text.str.split(expand=True).stack().value_counts().reset_index()
            new_df.columns = ['Word', 'Frequency'] 
            col1.write(new_df)
        
            st.download_button(label='Download Word Count' ,data=new_df.to_csv(),file_name='words.csv',mime='text/csv')      
                                  

            with col2:
                st.markdown('''
                # **Word Frequency Graph**
                ---
                ''')
                fig = px.bar(        
                new_df,
                x = "Frequency",
                y = "Word",
                color="Frequency", width=800, height=1000,
                orientation = 'h' #Optional Parameter
                )
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig)
            

            st.markdown('''
            # **Word Cloud**
            ---
            ''')

            from wordcloud import WordCloud 
            from matplotlib import colors
            from nltk.corpus import stopwords
            color_list=['#4d4d4d','#ffcc00','#000000',]
            colormap=colors.ListedColormap(color_list)
            import matplotlib.pyplot as plt
    
            new_df2 =  new_df.copy()

            mask = np.array(Image.open("bee.png"))
            words = ' '.join([Word for Word in new_df2['Word']])
            wordCloud = WordCloud(background_color=None, colormap='OrRd', min_font_size=4, max_font_size=100, width=3000, height=800, mask=mask, mode = 'RGB').generate(words)
            plt.figure(figsize=(50, 50), facecolor='k')
            plt.imshow(wordCloud, interpolation='bilinear')
            plt.axis("off")
            st.pyplot()
            
                    #####################################################################

            
def page8():
    st.markdown("# Sentiment Analyzer 🎉")
    st.sidebar.markdown("# Text Sentiments 🎉")    
        
  
    #adding a file uploader

    file = st.file_uploader("Please choose a file")

    if file is not None:
        df= pd.read_csv(file)
        #Can be used wherever a "file-like" object is accepted:

        df['Polarity'] = df.apply(lambda x: TextBlob(x['Cleaned_Text']).sentiment.polarity, axis=1)
        df['Subjectivity'] = df.apply(lambda x: TextBlob(x['Cleaned_Text']).sentiment.subjectivity, axis=1)
    
        def condition(x):
            if x>0.01:
                return "Positive"
            elif x<0:
                return "Negative"
            else:
                return 'Neutral'
    
        df['Sentiment'] = df['Polarity'].apply(condition)
    
        st.write(df)
        st.download_button(label='Download Sentiment Report ',data=df.to_csv(),file_name='sentiments.csv',mime='text/csv')

        col1, col2 = st.columns(2)
 
        with col1:    
            col1.header = "Sentiment"
            st.title = "Scatter"
            fig = px.scatter(df, x="Polarity", y="Subjectivity", color="Sentiment", width=750, height=700, opacity=0.7, color_discrete_sequence=["#FCE6C9","#00BFFF","#CD2626"])    
            fig.update_traces(marker=dict(size=20, line=dict(width=2,color='DarkSlateGrey')), selector=dict(mode='markers'))
            st.plotly_chart(fig)
    
            fig = px.histogram(df, x="Subjectivity")
            st.plotly_chart(fig)
        
            fig = px.histogram(df, x="Polarity")
            st.plotly_chart(fig)       
        
 
        with col2:
            col2.header = "Subjectivity"
            fig = px.pie(df, values='Subjectivity', names='Sentiment', color_discrete_sequence=["#00BFFF", "#CD2626", "#FCE6C9"])
            st.plotly_chart(fig) 
        
            from wordcloud import WordCloud 
            from matplotlib import colors
            from nltk.corpus import stopwords
            color_list=['#4d4d4d','#ffcc00','#000000',]
            colormap=colors.ListedColormap(color_list)
            import matplotlib.pyplot as plt
            st.set_option('deprecation.showPyplotGlobalUse', False)

            words = ' '.join([Text for Text in df[df['Sentiment']=='Negative']['Cleaned_Text']])
            wordCloud = WordCloud(background_color='black', colormap='Reds', min_font_size=4, max_font_size=200, width=3000, height=1000).generate(words)
            plt.figure(figsize=(50, 50))
            plt.imshow(wordCloud, interpolation='bilinear')
            plt.axis("off")
            st.pyplot() 
            st.set_option('deprecation.showPyplotGlobalUse', False)
        


            words = ' '.join([Text for Text in df[df['Sentiment']=='Positive']['Cleaned_Text']])
            wordCloud = WordCloud(background_color='black', colormap='Blues', min_font_size=4, max_font_size=200, width=3000, height=1000).generate(words)
            plt.figure(figsize=(50, 50))
            plt.imshow(wordCloud, interpolation='bilinear')
            plt.axis("off")
            st.pyplot()
            st.set_option('deprecation.showPyplotGlobalUse', False)
        
            from wordcloud import WordCloud 
            from matplotlib import colors
            from nltk.corpus import stopwords
            color_list=['#4d4d4d','#ffcc00','#000000',]
            colormap=colors.ListedColormap(color_list)
            import matplotlib.pyplot as plt

            words = ' '.join([Text for Text in df[df['Sentiment']=='Neutral']['Cleaned_Text']])
            wordCloud = WordCloud(background_color='black', colormap='YlOrBr', min_font_size=4, max_font_size=200, width=3000, height=1000).generate(words)
            plt.figure(figsize=(50, 50))
            plt.imshow(wordCloud, interpolation='bilinear')
            plt.axis("off")
            st.pyplot()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            
            senti = df['Sentiment'].value_counts()
            st.write(senti)
        
            color = ['bisque','deepskyblue','firebrick']
        
            fig = plt.figure(figsize=(10, 4))
            sns.set_style("darkgrid")
            sns.countplot(x = 'Sentiment', data=df, palette=color)
            st.pyplot(fig)
            
                    #####################################################################

    
    
def page9():
    st.markdown("# Counter 🎉")
    st.sidebar.markdown("# Count instances 🎉")       
    
    text_file = st.file_uploader("Upload CSV File",type=['csv'])

    if text_file is not None:
              
        col1, col2 = st.columns(2)
    
        with col1:
            df= pd.read_csv(text_file)
            st.write(df)
        
        with col2:
            df["Count"] = df['Text']
            new_df = df.Count.value_counts()
            st.write(new_df) 

page_names_to_funcs = {
    "Main Page": main_page,
    "Filter": page2,
    "Merge": page3,
    "EDA": page4,
    "Rename": page5,
    "Cleaner": page6,
    "Word Frequency": page7,
    "Sentiment Analyzer": page8,
    "Counter": page9
}

selected_page = st.sidebar.selectbox("Choose Here", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
