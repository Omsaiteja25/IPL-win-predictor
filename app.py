import streamlit as st
import pickle
import sklearn
import pandas as pd

teams = ['Sunrisers Hyderabad', 'Mumbai Indians',
         'Royal Challengers Bangalore', 'Kolkata Knight Riders',
         'Chennai Super Kings', 'Rajasthan Royals',
         'Kings XI Punjab', 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

pipe = pickle.load(open('new_pipe.pkl', 'rb'))
st.header('IPL Win Predictor 🏏')

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('select the batting team 🏏', sorted(teams))
with col2:
    bowling_team = st.selectbox('select the bowling team 🥎', sorted(teams))

col3, col4 = st.columns(2)

with col3:
    city = st.selectbox('Host City 🏙', sorted(cities))
with col4:
    target = st.number_input('Target 🎯')

col5, col6, col7 = st.columns(3)

with col5:
    score = st.number_input('Score')
with col6:
    overs = st.number_input('Overs Completed')
with col7:
    wickets = st.number_input('Wickets out ')

if st.button('Predict 🔍'):
    runs_left = target - score
    balls_left = 120 - (overs*6)
    wickets_left = 10 - wickets
    crr = score/overs
    rrr = (runs_left*6)/balls_left
    input_df = pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],'city':[city],
                             'runs_left':[runs_left], 'balls_left':[balls_left],'wickets':[wickets_left],
                            'total_runs_x':[target],'crr':[crr],'rrr':[rrr]})

    st.table(input_df)
    result = pipe.predict_proba(input_df)
    win = result[0][1]
    loss = result[0][0]

    st.header(batting_team + ' - ' + str(round(win*100)) + "%")
    st.header(bowling_team + ' - ' + str(round(loss*100)) + "%")



