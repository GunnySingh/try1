
from turtle import width
from matplotlib.ft2font import HORIZONTAL
from matplotlib.style import use
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import requests
from datetime import date
from datetime import datetime
import difflib
from streamlit_option_menu import option_menu
from PIL import Image
import collections
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import altair as alt
import re

st.set_page_config(page_title='Hi',page_icon='clapper')

st.markdown("""
<style>
footer{visibility:visible;}
footer:after{content:'Created by Gunny';
display:block;
color:tomato;
top:3px;}</style>""",unsafe_allow_html=True)

with st.sidebar:
    sel = option_menu(menu_title = 'Main Menu',options = ['Home','Multi-Select Movies','Year','Actors','Director','About'],\
        menu_icon=['list'],icons=['house','check-all','calendar-date','person-bounding-box','person-workspace','info-circle'],
        default_index=0)


sim_mat = pickle.load(open('sim_mat.pkl','rb'))
final = pickle.load(open('df_final.pkl','rb'))
actor = pickle.load(open('df_actor.pkl','rb'))

def recommend(movie):
    index = final[final.key == movie].index[0]
    results = sorted(list(enumerate(sim_mat[index])),key=lambda x :x[1])

    idx = []
    for i in results[:13]:
        idx.append(i[0])


    return final.iloc[idx]['title'].values[1:],'https://image.tmdb.org/t/p/w500'+final.iloc[idx]['poster'].values[1:],final.iloc[idx]['key'].values[1:]



if sel=='Home':
    
    # st.title('MOVIE RECOMMENDATION SYSTEM')
    st.markdown("""
    <h3 style = font-size:270%;text-align:center;color:darkslategray;>
    MOVIE RECOMMENDATION SYSTEM
    </h3>
    """,unsafe_allow_html=True)
    name = st.selectbox(label='Please select the Movie',options=final['key'],index=7421,help='Select Movie from below to get recommendations')

    # st.subheader(name)
    st.markdown("""
    <h2 style=color:teal;text-align:center;> {}</h1>""".format(name),unsafe_allow_html=True)

    index = final[final.key == name].index[0]
    poster_path = 'https://image.tmdb.org/t/p/w500'+final.iloc[index]['poster']
    year = final.iloc[index]['year']
    runtime = final.iloc[index]['runtime']
    rating = final.iloc[index]['rating']
    genre = final.iloc[index]['genre']
    summary = final.iloc[index]['overview']
    director = final.iloc[index]['director'] 
    cast = final.iloc[index]['cast']
    num_votes = final.iloc[index]['num_votes']
    budget = final.iloc[index]['budget']
    revenue = final.iloc[index]['revenue']

    col1 , col2 =st.columns(2)

    with col1:
        st.image(poster_path)

    with col2 :
        st.markdown('**`Name`** : {}'.format(final.iloc[index]['title']))
        st.markdown('**`Year`** : {}'.format(year))
        st.markdown('**`Runtime`** : {}'.format(runtime))
        st.markdown('**`Genre`** : {}'.format(genre))
        st.markdown('**`Rating`** : {} ({} Votes)'.format(rating,num_votes))
        st.markdown('**`Budget`** : {}'.format(budget))
        st.markdown('**`Revenue`** : {}'.format(revenue))
        st.markdown('**`Summary`** : {}'.format(summary))
        st.markdown('**`Director`** : {}'.format(director))
        

    len_cast = len(cast.split(','))
    cast_names = cast.split(',')


    def age_cast(birthdate):
        today = date.today()
        d = datetime.strptime(birthdate,"%Y-%m-%d")
        age = ((today-d.date())/365).days
        return str(age)+' Years'

    api_key = 'ef9ce1abb955e162c424955afe1df5a7'
    st.markdown('#### CAST :')

    for i,k in enumerate(st.columns(len_cast)):
        name_cast = cast_names[i]
        name_cast = difflib.get_close_matches(name_cast,actor.name)[0]
        cast_idx = actor[actor.name==name_cast].index[0]
        poster_path_cast = actor.iloc[cast_idx]['poster']
        k.image(poster_path_cast)
        if k.button(cast_names[i],key=i):
            
            cast_id = actor.iloc[cast_idx]['id']
            res = requests.get('https://api.themoviedb.org/3/person/{}?api_key={}&language=en-US'.format(cast_id,api_key))
            data = res.json()
            age = age_cast(data['birthday'])
            k.markdown('*Age* : {}'.format(age))
            k.markdown('*Born* : {}'.format(data['place_of_birth']))
            # k.markdown('*Biography* : {}'.format(data['biography']))
            st.text_area('Bio',data['biography'],height=200)
            st.markdown('*Best Known For:*')
            
            known_for_idx = []

            for i,k in enumerate(final.cast):
                for j in k.split(','):
                    if re.search(name_cast+'$',j):
                        known_for_idx.append(i)

            known_posters = final.iloc[known_for_idx].sort_values(by='num_votes_num',ascending=False)[:4]['poster'].values
            known_title = final.iloc[known_for_idx].sort_values(by='num_votes_num',ascending=False)[:4]['title'].values
            for i,cols in enumerate(st.columns(len(known_posters))):
                cols.image('https://image.tmdb.org/t/p/w500'+known_posters[i],caption=known_title[i])

            


    with st.sidebar:
        if st.button('Clear All Dropdowns'):
            st.write(' ')


    st.subheader('Recommendations For You :')


    titles , posters,title_key =  recommend(name)


    c=0
    for i in range(0,12,2):
        for col in st.columns(2):
            col.image(posters[c],width=250,caption=titles[c])
            
            # col.markdown('**{}**'.format(titles[c]))
            
            if col.button('Find Similar Movies',key=c):
                t,p,k = recommend(title_key[c])
                with st.expander('Recoms :',expanded=True):
                        w=0
                        for k in st.columns(5):
                            k.image(p[w],caption=t[w],width=80)
                            w+=1

                        # st.image(p[x],width=100,caption=t[x])
                        
            c+=1

if sel=='Year':
    st.markdown("""
    <h1 style ='color:DarkSlateGray;font-size:250%;text-align:center;'>
    SEARCH BY YEAR
    </h1>""",unsafe_allow_html=True)
    year_range = st.slider(label='Select Year',min_value=1950,max_value=2021,step=1,value=[2005,2021])

    eda_df = pickle.load(open('eda_df.pkl','rb'))
    start = year_range[0]
    end = year_range[1]
    eda= eda_df[eda_df.year.between(start,end,inclusive=True)]

    radio_sel = st.radio(label='',options=['No. Of Movies','Revenue By Year'],index=0,horizontal=True)
    if radio_sel =='No. Of Movies':
        a = px.bar(data_frame=eda, x='values',y='year',labels={'values':'No. Of Movies','year':'Year'},color='values',\
            height=1000,width=750, orientation='h',title='NO. OF MOVIES BY YEAR',color_continuous_scale='ice')
        a.update_yaxes(nticks=len(eda)+1)
        st.plotly_chart(a,use_container_width=False)

        if st.button(label='Raw Data'):
            st.dataframe(eda[['year','values']])

    if radio_sel == 'Revenue By Year':
        b = px.scatter(data_frame=eda,x='year',y='revenue',size='revenue',size_max=25,color='revenue',color_continuous_scale='brbg',\
            labels={'year':'YEAR','revenue':'REVENUE'},title='REVENUE BY YEAR',hover_name='title')    
        st.plotly_chart(b,use_container_width=True)

        if st.button(label='Raw Data'):
            st.dataframe(eda[['year','revenue','title']])

    
if sel =='Multi-Select Movies':

    st.markdown("""
    <h3 style='color:darkslategray;font-size:250%;text-align:center;'>
    MULTI-MOVIE RECOMMENDATION
    </h3>""",unsafe_allow_html=True)

    st.caption('You can get more accurate recommendations by selecting more than one movie.')
    st.write(' ')
    col1,col2 = st.columns([1,4])
    no_movies = col1.selectbox(label='No. of movies',options=range(2,9),index=2)
    multi_movies = col2.multiselect(label='Select your favoutive movies below :',options=final.key)
    
    multi_movies_idx=[]
    for i in multi_movies:
        multi_movies_idx.append(final[final.key==i].index[0])
    
    multi_movies_titles = final.iloc[multi_movies_idx]['title'].values
    multi_movies_ratings = final.iloc[multi_movies_idx]['rating'].values
    multi_movies_votes = final.iloc[multi_movies_idx]['num_votes'].values
    if len(multi_movies)==no_movies:

        if no_movies<=4:
            for i,col in enumerate(st.columns(no_movies)):
                col.image('https://image.tmdb.org/t/p/w500'+final[final.key==multi_movies[i]]['poster'].values[0],width=150,caption = multi_movies_titles[i]+' '+'\n'+str(multi_movies_ratings[i])+'('+str(multi_movies_votes[i])+' Votes)')

        if no_movies>4:
            for i,col in enumerate(st.columns(4)):
                col.image('https://image.tmdb.org/t/p/w500'+final[final.key==multi_movies[i]]['poster'].values[0],width=150,caption=multi_movies_titles[i]+' '+'\n'+str(multi_movies_ratings[i])+'('+str(multi_movies_votes[i])+' Votes)')
                
                
            for i,col in enumerate(st.columns(4)):
                try:
                    col.image('https://image.tmdb.org/t/p/w500'+final[final.key==multi_movies[4:][i]]['poster'].values[0],width=150,caption=multi_movies_titles[4:][i]+' '+'\n'+str(multi_movies_ratings[4:][i])+'('+str(multi_movies_votes[4:][i])+' Votes)')
                    
                except:
                    st.write('')
    
    # with st.expander(label='Get Reommendations:'):
    radio_sel = st.radio(label=' ',options=('Show Recommendations',"Don't Show Recommendations"),horizontal=True,index=1)
    if radio_sel=='Show Recommendations':
        s=np.zeros((11321,))
        for i in multi_movies_idx:
            s+=sim_mat[i]
        
        multi_results = sorted(list(enumerate(s)),key=lambda x :x[1])
        multi_results_idx = []
        for j in multi_results:
            multi_results_idx.append(j[0])

        for i in multi_movies_idx:
            multi_results_idx.remove(i)
        multi_results_idx=multi_results_idx[:12]
        multi_results_posters = final.iloc[multi_results_idx]['poster'].values
        multi_results_title = final.iloc[multi_results_idx]['title'].values
        multi_results_rating = final.iloc[multi_results_idx]['rating'].values
        multi_results_votes = final.iloc[multi_results_idx]['num_votes'].values
        multi_results_budget = final.iloc[multi_results_idx]['budget'].values
        multi_results_year = final.iloc[multi_results_idx]['year'].values
        multi_results_runtime = final.iloc[multi_results_idx]['runtime'].values
        multi_results_genre = final.iloc[multi_results_idx]['genre'].values
        multi_results_overview = final.iloc[multi_results_idx]['overview'].values
        multi_results_director = final.iloc[multi_results_idx]['director'].values
        multi_results_cast = final.iloc[multi_results_idx]['cast'].values
        multi_results_revenue = final.iloc[multi_results_idx]['revenue'].values
        
        
        q=0
        for i in range(0,12,3):
            for col in st.columns(3):
                col.image('https://image.tmdb.org/t/p/w500'+multi_results_posters[q],caption=multi_results_title[q])

                if col.button(label='Get Movie Info',key=q+100):
                    
                    col.markdown(' *Year:* {}'.format(multi_results_year[q]),unsafe_allow_html=True)
                    
                    col.markdown('_Runtime:_ {}'.format(multi_results_runtime[q]),unsafe_allow_html=True)
                    col.markdown('_Rating:_ {}({} Votes)'.format(multi_results_rating[q],multi_results_votes[q]),unsafe_allow_html=True)
                    col.markdown('_Genre:_ {}'.format(multi_results_genre[q]),unsafe_allow_html=True)
                    
                    col.markdown('_Director:_ {}'.format(multi_results_director[q]),unsafe_allow_html=True)
                    col.markdown('_Cast:_ {}'.format(multi_results_cast[q]),unsafe_allow_html=True)
                    col.markdown('_Budget:_ {}'.format(multi_results_budget[q]),unsafe_allow_html=True)
                    col.markdown('_Revenue:_ {}'.format(multi_results_revenue[q]),unsafe_allow_html=True)                    
                    st.text_area(label='Summary:',value= multi_results_overview[q])

                q+=1

if sel =='Actors':
    st.markdown("""
    <h1 style ='color:DarkSlateGray;font-size:250%;text-align:center;'>
    SEARCH BY ACTOR
    </h1>""",unsafe_allow_html=True)
    st.header('Coming Soon...')


if sel =='Director':
    st.markdown("""
    <h1 style ='color:DarkSlateGray;font-size:250%;text-align:center;'>
    SEARCH BY DIRECTOR
    </h1>""",unsafe_allow_html=True)

    st.header('Coming Soon...')


if sel == 'About':
    st.markdown("""
    <h1 style ='color:DarkSlateGray;font-size:250%;font-family:Times New Roman;text-align:center;'>
    ABOUT
    </h1>""",unsafe_allow_html=True)
    st.header('Coming Soon...')
