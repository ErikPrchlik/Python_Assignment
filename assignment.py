import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


def remove_duplicities(df, df_column):
    duplicate_ids = df.duplicated(subset=[df_column], keep=False)
    if duplicate_ids.any():
        print("Duplicate Product ID entries found:")
        print(df[duplicate_ids])
        df.drop(df[duplicate_ids].index, inplace=True)
    else:
        print("No duplicate Product ID entries found.")


def remove_missing(df, df_column):
    missing_values = df.isnull().any()
    if missing_values.any():
        print("Missing values found in the following columns:")
        print(missing_values)
        print(missing_values.sum())
    else:
        print("No missing values found in any columns.")

    missing_values = df.isnull().all()
    if missing_values.any():
        print("columns full of missing values found:")
        print(missing_values)
        print(missing_values.sum())
        df = df.dropna(axis=1, how='all')
    else:
        print("No missing values found in whole column for each.")

    missing_values = df[df_column].isnull()
    if missing_values.any():
        print("Missing values found in the id column:")
        print(missing_values.sum())
        print(df[missing_values])
        df.drop(df[missing_values].index, inplace=True)
    else:
        print("No missing values found in id column.")

    print("DataFrame after dropping columns and rows with all null values:")
    print(df)
    return df


def check_correct_digits(df, df_column):
    incorrect_types = ~df[df_column].apply(lambda x: str(x).replace('.', '').isdigit())
    if incorrect_types.any():
        print("Incorrect data types found in the column:")
        print(df[incorrect_types])
        if 'id' == df_column or 'external_id' == df_column:
            df.drop(df[incorrect_types].index, inplace=True)
        else:
            df[df_column].fillna(0, inplace=True)
    else:
        print("All values in the column have correct data types.")


def extract_number(s):
    try:
        return float(re.findall(r'\d+\.\d+', s)[0])
    except IndexError:
        return float('nan')


def df_columns_consistency(df):
    consistent_datatypes = df.dtypes.nunique() == 1
    if consistent_datatypes:
        print("All columns have consistent data types:", df.dtypes[0])
    else:
        print("Columns have inconsistent data types.")
        print("Data types for each column:")
        print(df.dtypes)
    print('####################################################################')


def load_data():
    df_available = pd.read_csv("market-units-available-anonymized.csv")
    df_sales = pd.read_csv("market-units-raw-sales-anonymized.csv")
    print('Data Loaded')
    return df_available, df_sales


def data_preprocessing(df_available, df_sales):
    remove_duplicities(df_available, 'id')
    remove_duplicities(df_sales, 'external_id')

    print('Duplicities removed')
    print('####################################################################')

    df_available = remove_missing(df_available, 'id')
    df_sales = remove_missing(df_sales, 'external_id')

    print('Missing values fixed')
    print('####################################################################')

    check_correct_digits(df_available, 'id')
    check_correct_digits(df_available, 'price')
    check_correct_digits(df_available, 'price_per_sm')
    check_correct_digits(df_available, 'total_area')
    check_correct_digits(df_available, 'developer_id')
    check_correct_digits(df_available, 'project_id')

    check_correct_digits(df_sales, 'price')
    check_correct_digits(df_sales, 'floor_area')
    df_sales['exterior_area'] = df_sales['exterior_area'].apply(extract_number)
    check_correct_digits(df_sales, 'exterior_area')

    print('Numerical columns checked')
    print('####################################################################')
    return df_available, df_sales


def correct_data_types(df_available, df_sales):
    df_available['id'] = df_available['id'].astype(int)
    df_available['cellar_included'] = df_available['cellar_included'].astype(bool)
    df_available['completed'] = df_available['completed'].astype(bool)
    df_available['parking_included'] = df_available['parking_included'].astype(bool)
    df_available['exterior_area'] = df_available['exterior_area'].apply(extract_number)
    df_available['exterior_area'] = df_available['exterior_area'].astype(float)
    df_available['last_seen'] = pd.to_datetime(df_available['last_seen'], format='%m/%d/%Y')
    df_available['first_seen'] = pd.to_datetime(df_available['first_seen'], format='%m/%d/%Y')
    df_available.loc[df_available['orientation'] == '-', 'orientation'] = np.nan

    df_sales['date_sold'] = pd.to_datetime(df_sales['date_sold'], format='%m/%d/%Y %H:%M')
    df_sales['last_seen'] = pd.to_datetime(df_sales['last_seen'], format='%m/%d/%Y')
    df_sales['first_seen'] = pd.to_datetime(df_sales['first_seen'], format='%m/%d/%Y')
    df_sales['price'] = df_sales['price'].astype(float)
    df_sales['exterior_area'] = df_sales['exterior_area'].astype(float)

    print('Data types corrected')
    print('####################################################################')
    return df_available, df_sales


def drop_useless_columns(df_available, df_sales):
    df_available.drop(columns=['cellar_price_per_sm', 'balcony_area', 'bath_area', 'bath_full_area', 'bath_half_area', 'bedroom_area', 
        'city', 'corridor_area', 'date_added_numeric', 'date_reserved_numeric', 'date_sold_numeric', 'dining_area', 'entity_buyer', 
        'entry_area', 'terrace_area', 'wardrobe_area', 'stairs_area', 'storage_area', 'months_till_completed', 'permit_regular', 
        'price_real', 'price_per_sm_real', 'building_id', 'garden_area', 'internal_buyer', 'kitchen_area', 'lee', 'living_area', 
        'living_room_area', 'parking_indoor_price', 'parking_outdoor_price', 'payment_construction', 'payment_contract', 
        'payment_occupancy'], inplace=True)

    df_sales.drop(columns=['garden_area', 'terrace_area', 'balcony_area'], inplace=True)

    print('Useless columns droped')
    print('####################################################################')
    return df_available, df_sales


def top_developer(df_sales):
    developer_sales = df_sales.groupby('developer_name').size().reset_index(name='total_units_sold').sort_values(by='total_units_sold', ascending=False)
    top_developer = developer_sales.loc[developer_sales['total_units_sold'].idxmax()]
    print("Top Developer Summary:")
    print(f"Developer Name: {top_developer['developer_name']}")
    print(f"Total Units Sold: {top_developer['total_units_sold']}")

    print("#########################")

    print("Top 5 Developers Summary:")
    for i in range(5):
        developer_info = developer_sales.iloc[i]
        print(f"{i+1}. Developer Name: {developer_info['developer_name']}")
        print(f"   Total Units Sold: {developer_info['total_units_sold']}")

    print("#########################")
    print(f"Totaly Sold Units: {developer_sales['total_units_sold'].sum()}")
    print("#########################")


def sales_statistics(df_sales):
    df_sales['year_sold'] = df_sales['date_sold'].dt.year
    df_sales['month_sold'] = df_sales['date_sold'].dt.month
    sales_statistics = df_sales.groupby(['year_sold', 'developer_name']).size().reset_index(name='total_units_sold').sort_values(by=['year_sold','total_units_sold'], ascending=False)
    print("Statistics of Sold Units by Year and Developer Name:")
    print(sales_statistics)

    print("#########################")

    top_developers_per_year = []
    for year in sales_statistics['year_sold'].unique():
        top_developers_year = sales_statistics[sales_statistics['year_sold'] == year].head(5)
        top_developers_per_year.append(top_developers_year)

    result_df = pd.concat(top_developers_per_year, ignore_index=True)
    print("Statistics of Top 5 Developers per Year:")
    print(result_df)

    print("#########################")


def project_sales(df_sales):
    project_sales = df_sales.groupby('project_name').size().reset_index(name='total_units_sold').sort_values(by='total_units_sold', ascending=False)
    print("Top 5 Projects Summary:")
    for i in range(5):
        project_info = project_sales.iloc[i]
        print(f"{i+1}. Developer Name: {project_info['project_name']}")
        print(f"   Total Units Sold: {project_info['total_units_sold']}")

    print("#########################")


def average_price(df_sales):
    average_price_per_developer = df_sales.groupby('developer_name')['price'].mean().reset_index(name='average_price').sort_values(by='average_price', ascending=False)
    average_price_per_developer['average_price'] = average_price_per_developer['average_price'].apply(lambda x: '{:,.2f}'.format(x))
    print("Average Selling Price per Developer:")
    print(average_price_per_developer)

    print("#########################")

    average_price_per_project = df_sales.groupby('project_name')['price'].mean().reset_index(name='average_price').sort_values(by='average_price', ascending=False)
    average_price_per_project['average_price'] = average_price_per_project['average_price'].apply(lambda x: '{:,.2f}'.format(x))
    print("Average Selling Price per Project:")
    print(average_price_per_project)

    print("#########################")


def sales_velocity(df_sales):
    df_sales['sale_duration'] = df_sales['date_sold'] - df_sales['first_seen']

    average_sales_velocity_per_developer = df_sales.groupby('developer_name').agg(
        average_sales_velocity=('sale_duration', 'mean'),
        num_units_sold=('date_sold', 'size')
    ).reset_index().sort_values(by=['average_sales_velocity', 'num_units_sold'], ascending=[True, False])

    print("Average Sales Velocity per Developer:")
    print(average_sales_velocity_per_developer)


def price_consistancy(df_available):
    # Calculate price per square meter from total area and floor area
    df_available['price_per_sm_total_area'] = df_available['price'] / df_available['total_area']
    df_available['price_per_sm_floor_area'] = df_available['price'] / df_available['floor_area']

    # Check if price per square meter matches with either total area or floor area
    df_available['is_price_per_sm_total_area'] = df_available['price_per_sm'] == df_available['price_per_sm_total_area']
    df_available['is_price_per_sm_floor_area'] = df_available['price_per_sm'] == df_available['price_per_sm_floor_area']

    # Print the DataFrame filtered by the two conditions
    print("All price per sm matches with total area:")
    print(df_available['is_price_per_sm_total_area'].all())

    print("All price per sm matches with floor area:")
    print(df_available['is_price_per_sm_floor_area'].all())
    print('####################################################################')


def price_diff_per_building(df_available):
    # Group the DataFrame by the 'building' column and calculate average price
    building_avg_price = df_available.groupby('building')['price'].mean()

    # Calculate the price difference between units within each building
    df_available['price_difference'] = df_available.groupby('building')['price'].transform(lambda x: x - x.mean())
    building_avg_price_diff = df_available.groupby('building')['price_difference'].mean()

    # Find the building with the largest average price difference
    building_with_max_price_diff = building_avg_price_diff.idxmax()

    # Print all units of the building with the biggest average price difference
    units_of_max_price_diff_building = df_available[df_available['building'] == building_with_max_price_diff]
    df_columns_consistency(units_of_max_price_diff_building)
    print("Units of the building with the biggest average price difference:")
    print(units_of_max_price_diff_building[['availability', 'building', 'floor', 'layout', 'total_area', 'floor_area', 'exterior_area', 'price', 'price_per_sm', 'price_difference']])
    print('####################################################################')


def analysis_df_available(df_available):
    price_consistancy(df_available)
    price_diff_per_building


def data_analysis(df_available, df_sales):
    print('####################################################################')
    print('Statistical Analysis')
    print('####################################################################')
    top_developer(df_sales)
    sales_statistics(df_sales)
    project_sales(df_sales)
    average_price(df_sales)
    sales_velocity(df_sales)

    analysis_df_available(df_available)


def plot_layout_distribution(df_available, df_sales):
    df_filtered = df_sales.dropna(subset=['layout']).copy()
    layout_mapping = {
        np.nan: 0,
        'LAYOUT_1': 1,
        'LAYOUT_1_5': 1.5,
        'LAYOUT_2': 2,
        'LAYOUT_3': 3,
        'LAYOUT_4': 4,
        'LAYOUT_5': 5,
        'LAYOUT_6': 6,
        'LAYOUT_7': 7
    }
    df_filtered['num_rooms'] = df_filtered['layout'].map(layout_mapping)
    df_sales['num_rooms'] = df_sales['layout'].map(layout_mapping)
    plt.figure(figsize=(10, 6))
    sns.countplot(x='num_rooms', data=df_filtered)
    plt.title('Distribution of Properties by Number of Rooms')
    plt.xlabel('Number of Rooms')
    plt.ylabel('Number of Properties')
    plt.xticks(rotation=45)
    plt.tight_layout()


def plot_top_projects(df_available, df_sales):
    project_sales = df_sales.groupby('project_name').size().reset_index(name='total_units_sold').sort_values(by='total_units_sold', ascending=False)
    top_5_projects = project_sales.head(5)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='total_units_sold', y='project_name', data=top_5_projects, palette='viridis')
    plt.title('Sales Performance of Top 5 Projects')
    plt.xlabel('Total Units Sold')
    plt.ylabel('Project Name')
    plt.tight_layout()


def plot_floor_area_distrubution(df_available, df_sales):
    plt.figure(figsize=(10, 6))
    df_sales['floor_area'].plot(kind='hist', bins=20, title='floor_area')
    plt.gca().spines[['top', 'right',]].set_visible(False)
    plt.title('Floor area Distribution')
    plt.xlabel('Meters per square')
    plt.ylabel('Frequency')
    plt.tight_layout()


def plot_month_year_sold(df_available, df_sales):
    df_sales['month_year_sold'] = df_sales['date_sold'].dt.to_period('M')
    units_sold_by_month = df_sales.groupby('month_year_sold').size()
    plt.figure(figsize=(10, 6))
    units_sold_by_month.plot(kind='bar', color='skyblue')
    plt.title('Count of Units Sold by Month')
    plt.xlabel('Month')
    plt.ylabel('Count of Units Sold')
    plt.xticks(rotation=45)
    plt.tight_layout()


def data_visualization(df_available, df_sales):
    print('####################################################################')
    print('Data Visualization')
    print('####################################################################')

    plot_layout_distribution(df_available, df_sales)
    plot_top_projects(df_available, df_sales)
    plot_floor_area_distrubution(df_available, df_sales)
    plot_month_year_sold(df_available, df_sales)

    plt.show()


def correlation(df_available, df_sales):
    # Calculate the correlation matrix
    df_filtered = df_available.copy()  # Make a copy to avoid the SettingWithCopyWarning
    layout_mapping = {
        np.nan: 0,
        'LAYOUT_1': 1,
        'LAYOUT_1_5': 1.5,
        'LAYOUT_2': 2,
        'LAYOUT_3': 3,
        'LAYOUT_4': 4,
        'LAYOUT_5': 5,
        'LAYOUT_6': 6,
        'LAYOUT_7': 7
    }
    df_filtered['num_rooms'] = df_filtered['layout'].map(layout_mapping)
    correlation_matrix = df_filtered[['price', 'price_per_sm', 'floor_area', 'total_area', 'floor', 'num_rooms', 'DOM', 'exterior_area']].corr()

    print('Legend of correlation:')
    print(' 1 indicates a perfect positive correlation')
    print('-1 indicates a perfect negative correlation')
    print(' 0 indicates no correlation')
    print("#########################")

    # Print the correlation matrix
    print("Correlation Matrix of df_available:")
    print(correlation_matrix)

    print("#########################")
    df_filtered = df_sales.copy()
    df_filtered['num_rooms'] = df_filtered['layout'].map(layout_mapping)
    correlation_matrix = df_filtered[['price', 'floor_area', 'exterior_area', 'num_rooms']].corr()

    # Print the correlation matrix
    print("Correlation Matrix of df_sales:")
    print(correlation_matrix)
    print('####################################################################')


def linear_regression(df_available, df_sales):
    X = df_sales[['floor_area', 'exterior_area']]
    X = sm.add_constant(X)
    X = sm.add_constant(X)
    y = df_sales['price']
    model = sm.OLS(y, X).fit()
    print(model.summary())

    print("#########################")

    p_values = model.pvalues.apply(lambda x: '{:,.5f}'.format(x))
    print("Low p-value (typically < 0.05): Indicates that the feature is likely to have a significant impact on the target variable.")
    print("Coefficients and p-values:")
    for feature, p_value in zip(X.columns, p_values):
        print(f"{feature}: P>|t|={p_value}")

    print("#########################")
    # Get the coefficients from the model summary
    coefficients = model.params
    # Create a DataFrame to store the coefficients
    coefficients_df = pd.DataFrame(coefficients, columns=['Coefficient'])
    # Sort the DataFrame by the absolute value of the coefficients to identify the most important parameters
    coefficients_df['Absolute_Coefficient'] = coefficients_df['Coefficient'].abs()
    coefficients_df = coefficients_df.sort_values(by='Absolute_Coefficient', ascending=False)
    # Print the most important parameters
    print("Impact of features on price:")
    print(coefficients_df)

    print("#########################")
    # Calculate average price per square meter
    # Filter out rows with zero or very small floor areas
    df_sales_filtered = df_sales[df_sales['floor_area'] > 0.1].copy()  # Adjust the threshold as needed
    # Calculate average price per square meter
    df_sales_filtered['average_price_per_sm'] = df_sales_filtered['price'] / df_sales_filtered['floor_area']
    # Print the average price per square meter
    print("Average Price per Square Meter of Floor area:")
    print(df_sales_filtered['average_price_per_sm'].mean())


def statistical_analysis(df_available, df_sales):
    print('####################################################################')
    print('Statistical Analysis')
    print('####################################################################')

    correlation(df_available, df_sales)

    linear_regression(df_available, df_sales)


def main():
    df_available, df_sales = load_data()
    df_available, df_sales = data_preprocessing(df_available, df_sales)
    df_available, df_sales = correct_data_types(df_available, df_sales)
    df_available, df_sales = drop_useless_columns(df_available, df_sales)
    print('DataFrame df_available')
    df_columns_consistency(df_available)
    print('DataFrame df_sales')
    df_columns_consistency(df_sales)

    data_analysis(df_available, df_sales)
    data_visualization(df_available, df_sales)
    statistical_analysis(df_available, df_sales)


if __name__ == "__main__":
    main()
