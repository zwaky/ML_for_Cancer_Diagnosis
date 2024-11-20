from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def plot_relationships(df):

    # Normalize the numerical columns except 'diagnosis'
    columns_to_plot = [col for col in df.columns if col != 'diagnosis']

    # Determine the number of plots
    num_columns = len(columns_to_plot)

    # Create subplots: adjust rows and columns for better layout
    rows = 6
    cols = (num_columns + rows - 1) // rows  # Round up to fit all plots
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), sharey=True)
    axes = axes.flatten()

    # Loop through the columns and create KDE plots
    for i, category in enumerate(columns_to_plot):
        sns.kdeplot(data=df, x=category, hue='diagnosis', fill=True, ax=axes[i])
        axes[i].set_title(f'{category} by Diagnosis', fontsize=10)
        axes[i].set_xlabel(category)
        axes[i].set_ylabel('Density')

    # Hide unused subplots if columns don't fill all grid spaces
    for ax in axes[num_columns:]:
        ax.axis('off')

    # Adjust layout
    plt.tight_layout()

    plt.show()

def main():

    df = pd.read_csv("data.csv")
    # print(pd.head())

    # Remove the last column
    if df.columns[-1].startswith("Unnamed"):
            df = df.iloc[:, :-1]  # Removes the last column

    # Extract target variable
    target = df.pop('diagnosis')

    # Convert results to 0 or 1
    target.replace("M", 1, inplace=True)
    target.replace("B", 0, inplace=True)

    # Return diagnosis to the data
    df['diagnosis'] = target

    #Normalize the data
    scale = StandardScaler()
    df_sc = scale.fit_transform(df)
    df_sc = pd.DataFrame(df_sc, columns=df.columns)
    df_sc['diagnosis'] = df['diagnosis']


    # TODO clean up data
    # # Setup data to recognize 0s as NAN values
    # df.replace(0, np.nan, inplace=True)
    # # check for NAN
    # print(df.isna().sum())

    #Showing how balanced the results are (relatively balanced in this case)
    # print(pd.crosstab(target,target,normalize='all')*100)


    y = df_sc['diagnosis']
    x = df_sc.drop('diagnosis', axis = 1)
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=1000)
    SVM_classification = SVC()
    SVM_classification.fit(x_train, y_train)

    y_hat = SVM_classification.predict(x_test)
    predictions = pd.DataFrame({'y_test': y_test, 'y_hat': y_hat})
    predictions.tail(20)


    print(classification_report(y_test, y_hat))



if __name__ == "__main__":
    main()
