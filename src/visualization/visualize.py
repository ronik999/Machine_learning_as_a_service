import seaborn as sns
import matplotlib.pyplot as plt

def get_bar_plot(df,category):
    max_category_sales = sns.barplot(data=df.groupby(['cat_id'])['revenue'].sum().reset_index(),
                                     x='cat_id', y='revenue')
    plt.title("MAX SALES CATEGORY")
    plt.show()

def get_actual_vs_predicted_plot(y_test, y_pred):

    sns.set(style="whitegrid")

    sns.scatterplot(x=y_test['revenue'], y=y_pred)
    sns.scatterplot(x=y_test['revenue'], y=y_test['revenue'])

    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs. Predicted Values')

    plt.show()

def feature_importance_plot(model):
    feature_importances = model.feature_importances_

    # Get the names of your features (if available)
    feature_names = ['item_id', 'cat_id', 'store_id', 'state_id', 'date', 'day', 'month']  # Your feature names

    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_importances)), feature_importances, tick_label=feature_names)
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.title('Feature Importance Plot')
    plt.show()


def plot_time_series_graph(df, col_x, col_y):
    sns.set_style('darkgrid')
    plt.title('REVENUE THROUGHOUT THE DATA')
    line_plt = sns.lineplot(data=df, x=df[col_x], y=df[col_y])
    plt.show()