# Virtual Used Car Salesman
#### by Shawn Oppermann

## Report Contents

- Overview
- Data Retrieval
- Python Environment/Libraries
- Exploring the Data
- Preparing the Data
- Modeling
- Fine Tuning
- Results and Conclusion

## Overview

Many people like myself don't have a lot of experience with buying and selling used cars. This project aims to create a model that can predict the price of a used car to help people figure out what's a reasonable price to buy and sell. This project uses regression models such as Ridge and Lasso to predict the price.

## Goal

The goal of this project is to create a regression model that can accurately predict the price of a car given the data as detailed in Data Contents, ideally within an absolute mean error of 500 dollars.

## Data

The data as available for download at https://www.kaggle.com/austinreese/craigslist-carstrucks-data#.

### Data Contents
   * vehicles.csv - dataset that includes vehicles posted on Craigslist in the year 2020
   
- The data is composed of 418213 rows and 26 columns.

I will only be focusing on 15 of these columns:
- price: the price the car goes for, our target variable as an integer
- year: the year of the car model as an integer
- manufacturer: the name of the manufacturer of the car
- model: the model name of the car
- condition: the condition of the car, ranging from brand new to salvaged
- cylinders: the number of cylinders in the car, treated as a qualitative column
- fuel: the type of fuel the car needs
- odometer: the number of miles traveled as read on the car's odometer.
- title_status: the status of the vehicle's title, often clean, lien, or missing
- transmission: manual or auto
- drive: real wheel drive or 4 wheel drive
- size: the size of the vehicle, often full-size or other
- type: the type of car; sedan, SUV, pickup, etc.
- paint_color: color of the car
- state: the state the car is being sold from

## Project Specifications

### Publishing
    - Contributor: Shawn Oppermann
    - Start date: 12/4/20

### Usage
    - Language: Python
    - Tools/IDE: Anaconda, Jupyter Notebook
    - Libraries: pandas, numpy, matplotlib, category_encoders, sklearn

