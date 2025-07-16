import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Load data
players = pd.read_csv('Scrapping/data/players.csv')
icons = pd.read_csv('Scrapping/data/Icons.csv')

def clean_players(df):
    df.rename(columns={'In Game Stats': 'IGS'}, inplace=True)
    df.drop(columns='Side Position', axis=1, inplace=True)
    df['Position'] = df['Position'].str.replace('++', '', regex=False)
    df['Popularity'] = df['Popularity'].str.replace(',', '').astype(float)
    df[['Height (cm)', 'Height (ft)', 'Body Description']] = df['Body Type'].str.extract(
        r'(\d+\s*cm)\s*\|\s*([\d\'\"]+)\s*\n*(.*)'
    )
    df['Body Description'] = df['Body Description'].str.replace(r'\s*\(.*?\)', '', regex=True)
    df.drop(columns=['Body Type', 'Height (ft)'], axis=1, inplace=True)
    df['Weight'] = df['Weight'].str.replace('kg', '', regex=False).astype(int)
    df['Price'] = df['Price'].apply(lambda x: float(x.replace('K', '')) * 1000 if 'K' in x else float(x))
    df['Height (cm)'] = df['Height (cm)'].str.replace('cm', '', regex=False).astype(int)
    df['Value for Money'] = (df['Rating'] * 1000) / df['Price']
    return df

def clean_icons(df):
    df.drop(index=259, inplace=True)
    df.rename(columns={'Player': 'Name'}, inplace=True)
    df.rename(columns={'Physic': 'Physicality'}, inplace=True)
    df.drop(columns='sidePosition', inplace=True)
    df['Position'] = df['Position'].str.replace('++', '', regex=False)
    df['Popularity'] = df['Popularity'].str.replace(',', '').astype(float)
    df['Weight'] = df['Weight'].str.replace('kg', '', regex=False).astype(int)
    df[['Height (cm)', 'Height (ft)', 'Body']] = df['Body'].str.extract(
        r'(\d+\s*cm)\s*\|\s*([\d\'\"]+)\s*\n*(.*)'
    )
    df.drop(columns=['Body','Height (ft)','Club','League'], inplace=True)
    df['Height (cm)'] = df['Height (cm)'].str.replace('cm', '', regex=False).astype(int)

    def replace(price):
        if 'K' in price:
            return float(price.replace('K', '')) * 1000
        elif 'M' in price:
            return float(price.replace('M', '')) * 1000000
        else:
            return float(price)

    df['Price'] = df['Price'].apply(replace)
    df.index = df['Name']
    df.loc['Edson Arantes Nascimento','Name'] = 'Pelé'
    df.reset_index(drop=True, inplace=True)
    df['Value for Money'] = (df['Rating'] * 1000) / df['Price']
    return df

players = clean_players(players)
icons = clean_icons(icons)

def generate_comparison_radar(player1, player2, df):
    attributes = ['Pace', 'Shooting', 'Passing', 'Dribbling', 'Defending', 'Physicality']
    values1 = df[df['Name'] == player1][attributes].values[0].tolist() + [df[df['Name'] == player1][attributes].values[0][0]]
    values2 = df[df['Name'] == player2][attributes].values[0].tolist() + [df[df['Name'] == player2][attributes].values[0][0]]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values1,
        theta=attributes + [attributes[0]],
        fill='toself',
        name=player1,
        line_color='blue'
    ))
    fig.add_trace(go.Scatterpolar(
        r=values2,
        theta=attributes + [attributes[0]],
        fill='toself',
        name=player2,
        line_color='red'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        title=f"Comparison: {player1} vs {player2}",
        height=500
    )
    return fig

def generate_position_heatmap(df):
    position_stats = df.groupby('Position')[['Pace', 'Shooting', 'Passing', 'Dribbling', 'Defending', 'Physicality']].mean()
    fig = px.imshow(position_stats, 
                   labels=dict(x="Attribute", y="Position", color="Average Rating"),
                   x=position_stats.columns,
                   y=position_stats.index,
                   aspect="auto",
                   color_continuous_scale='Viridis')
    fig.update_layout(title='Average Attributes by Position')
    return fig

def generate_top_players_table(df, attribute, n=10):
    top_players = df.sort_values(by=attribute, ascending=False).head(n)
    columns_to_show = ['Name', 'Position', 'Rating', 'Price', attribute]
    seen = set()
    unique_columns = [col for col in columns_to_show if not (col in seen or seen.add(col))]
    
    result = top_players[unique_columns].copy()
    if 'Price' in result.columns:
        result['Price'] = result['Price'].apply(lambda x: f"€{x/1000000:.2f}M" if x >= 1000000 else f"€{x/1000:.1f}K")
    return result

def format_player_summary(df, attribute):
    result = df[df[attribute] == df[attribute].max()][["Name", 'Position', 'Rating', 'Price', attribute]].copy()
    if 'Price' in result.columns:
        result['Price'] = result['Price'].apply(lambda x: f"€{x/1000000:.2f}M" if x >= 1000000 else f"€{x/1000:.1f}K")
    return result

def main():
    st.set_page_config(layout="wide", page_title="FIFA Player Analysis")
    
    st.sidebar.title("Filters")
    kind = st.sidebar.selectbox("Select Player Type", options=['Icons', 'Normal'], index=0)
    
    if kind == 'Icons':
        df = icons
    else:
        df = players
    
    # Add more filters
    min_rating = st.sidebar.slider("Minimum Rating", int(df['Rating'].min()), int(df['Rating'].max()), 80)
    max_price = st.sidebar.slider("Maximum Price (millions)", 
                                float(df['Price'].min()/1000000), 
                                float(df['Price'].max()/1000000), 
                                float(df['Price'].max()/1000000))
    
    position_filter = st.sidebar.multiselect("Filter by Position", 
                                           options=sorted(df['Position'].unique()),
                                           default=sorted(df['Position'].unique()))
    
    foot_filter = st.sidebar.multiselect("Filter by Strong Foot", 
                                       options=sorted(df['Strong Foot'].unique()),
                                       default=sorted(df['Strong Foot'].unique()))
    
    # Apply filters
    filtered_df = df[
        (df['Rating'] >= min_rating) &
        (df['Price'] <= max_price * 1000000) &
        (df['Position'].isin(position_filter)) &
        (df['Strong Foot'].isin(foot_filter))
    ].copy()
    
    st.title("FIFA Player Analysis Dashboard")
    
    # Key metrics with error handling
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Players", len(filtered_df))
    
    try:
        col2.metric("Average Rating", f"{filtered_df['Rating'].mean():.1f}")
        col3.metric("Average Price", f"€{filtered_df['Price'].mean()/1000000:.2f}M")
        
        # Handle empty mode case
        mode_positions = filtered_df['Position'].mode()
        most_common_pos = mode_positions[0] if not mode_positions.empty else "N/A"
        col4.metric("Most Common Position", most_common_pos)
    except Exception as e:
        st.error(f"Error calculating metrics: {str(e)}")
    
    if kind == 'Normal':
        # Club word cloud
        st.subheader("Club Distribution Word Cloud")
        try:
            text = ' '.join(filtered_df['Club'].str.replace(' ', '_', regex=False))
            wordcloud = WordCloud(width=800, height=400, background_color='white', min_font_size=10).generate(text)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Could not generate word cloud: {str(e)}")
        
        # League distribution
        st.subheader("League Distribution")
        try:
            top_leagues = ['LALIGA EA SPORTS','Premier League','Bundesliga','Serie A TIM','Ligue 1 Uber Eats']
            filtered_df['Top_Leagues'] = filtered_df['League'].apply(lambda x: 'other' if x not in top_leagues else x)
            league_counts = filtered_df['Top_Leagues'].value_counts().reset_index()
            league_counts.columns = ['League', 'Count']
            
            fig = px.pie(league_counts, values='Count', names='League', 
                        title='Distribution of Top Leagues', hole=0.3)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not generate league distribution: {str(e)}")
    
    # Top players by attributes
    st.subheader("Top Players by Attributes")
    try:
        attribute = st.selectbox("Select Attribute", 
                               options=['Rating', 'Pace', 'Shooting', 'Passing', 
                                      'Dribbling', 'Defending', 'Physicality', 
                                      'Value for Money', 'Popularity'])
        
        top_players_df = generate_top_players_table(filtered_df, attribute)
        st.dataframe(top_players_df,
                    height=400, 
                    use_container_width=True)
    except Exception as e:
        st.error(f"Error displaying top players: {str(e)}")
        if not filtered_df.empty:
            st.write(filtered_df.sort_values(by='Rating', ascending=False).head(10))
    
    # Position heatmap
    st.subheader("Position Attribute Heatmap")
    try:
        st.plotly_chart(generate_position_heatmap(filtered_df), use_container_width=True)
    except Exception as e:
        st.error(f"Could not generate position heatmap: {str(e)}")
    
    # Player comparison
    st.subheader("Player Comparison")
    try:
        col1, col2 = st.columns(2)
        with col1:
            player1 = st.selectbox("Select Player 1", options=sorted(filtered_df['Name'].unique()))
        with col2:
            player2 = st.selectbox("Select Player 2", 
                                  options=sorted(filtered_df['Name'].unique()),
                                  index=1 if len(filtered_df) > 1 else 0)
        
        st.plotly_chart(generate_comparison_radar(player1, player2, filtered_df), 
                       use_container_width=True)
    except Exception as e:
        st.error(f"Could not generate player comparison: {str(e)}")
    
    # Player summaries with error handling
    st.header("Player Summaries")
    
    attributes_to_summarize = [
        ('Pace', 'Fastest Players'),
        ('Shooting', 'Highest Shooting Players'),
        ('Passing', 'Highest Passing Players'),
        ('Dribbling', 'Highest Dribbling Players'),
        ('Defending', 'Highest Defending Players'),
        ('Physicality', 'Highest Physicality Players'),
        ('Popularity', 'Most Popular Players'),
        ('Price', 'Most Expensive Players'),
        ('Height (cm)', 'Tallest Players'),
        ('Height (cm)', 'Shortest Players'),
        ('Weight', 'Heaviest Players'),
        ('Weight', 'Lightest Players'),
        ('IGS', 'Highest In Game Stats Players')
    ]
    
    for attribute, title in attributes_to_summarize:
        try:
            if attribute in filtered_df.columns and not filtered_df.empty:
                st.subheader(title)
                st.dataframe(format_player_summary(filtered_df, attribute), use_container_width=True)
        except Exception as e:
            st.error(f"Could not display {title}: {str(e)}")
    
    # Advanced analytics section with error handling
    st.header("Advanced Analytics")
    
    # Price vs. Rating without trendline (removed LOWESS requirement)
    st.subheader("Price vs. Rating Analysis")
    try:
        fig = px.scatter(filtered_df, x='Rating', y='Price', 
                        color='Position', hover_name='Name',
                        title='Price vs. Rating with Position Coloring')
        fig.update_layout(xaxis_title='Rating', yaxis_title='Price (€)')
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Could not generate price vs rating plot: {str(e)}")
    
    # Age distribution
    st.subheader("Age Distribution")
    try:
        fig = px.histogram(filtered_df, x='Age', nbins=20, 
                          title='Distribution of Player Ages')
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Could not generate age distribution: {str(e)}")
    
    # Value for money analysis
    st.subheader("Best Value Players (High Rating, Low Price)")
    try:
        fig = px.scatter(filtered_df, x='Price', y='Rating', 
                        hover_name='Name', color='Position',
                        size='Value for Money',
                        title='Value for Money Analysis (Bigger circles = better value)')
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Could not generate value analysis: {str(e)}")
    
    # Physical attributes analysis
    st.subheader("Physical Attributes Analysis")
    try:
        fig = px.scatter_3d(filtered_df, x='Height (cm)', y='Weight', z='Physicality',
                           color='Position', hover_name='Name',
                           title='Height vs. Weight vs. Physicality')
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Could not generate physical attributes plot: {str(e)}")
    
    # Nation distribution
    st.subheader("Nationality Distribution")
    try:
        top_nations = filtered_df['Nation'].value_counts().nlargest(10).index
        filtered_df['Top_Nations'] = filtered_df['Nation'].apply(lambda x: x if x in top_nations else 'Other')
        nation_counts = filtered_df['Top_Nations'].value_counts()
        
        fig = px.bar(nation_counts, 
                    labels={'index': 'Nation', 'value': 'Count'},
                    title='Top 10 Nations by Player Count')
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Could not generate nationality distribution: {str(e)}")
    
    # Download button
    try:
        st.sidebar.download_button(
            label="Download Filtered Data",
            data=filtered_df.to_csv().encode('utf-8'),
            file_name='fifa_filtered_players.csv',
            mime='text/csv'
        )
    except Exception as e:
        st.sidebar.error("Could not prepare download data")

if __name__ == "__main__":
    main()

#streamlit run .\sreamlit_Futbin.py