{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "model_selection by lazypredict",
      "provenance": [],
      "authorship_tag": "ABX9TyMTv3Z64yp8x69aZQ8rWwQa"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "6-8vogY3s0Y_"
      },
      "outputs": [],
      "source": [
        "!pip install lazypredict\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import lazypredict\n"
      ],
      "metadata": {
        "id": "6OkzTBO1s_nG"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from lazypredict.Supervised import LazyClassifier\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "rjzIhAGHtBoF",
        "outputId": "c670c0e5-9d82-451d-b6bf-e0b40d2abf38"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/sklearn/utils/deprecation.py:143: FutureWarning: The sklearn.utils.testing module is  deprecated in version 0.22 and will be removed in version 0.24. The corresponding classes / functions should instead be imported from sklearn.utils. Anything that cannot be imported from sklearn.utils is now part of the private API.\n",
            "  warnings.warn(message, FutureWarning)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "import numpy as np"
      ],
      "metadata": {
        "id": "OV4v0Fh0toz5"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train16 = pd.read_csv('/content/train_mfcc_data(16).csv')\n",
        "train32 = pd.read_csv('/content/train_mfcc_data(32).csv')\n",
        "train64 = pd.read_csv('/content/train_mfcc_data(64).csv')"
      ],
      "metadata": {
        "id": "7XBEGzUltmzv"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "\n",
        "\n",
        "X = train16.drop(['covid19'], axis=1)\n",
        "y= train16['covid19']\n",
        "\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.2,random_state =123)\n",
        "\n",
        "clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)\n",
        "models,predictions = clf.fit(X_train, X_test, y_train, y_test)\n",
        "\n",
        "print(models)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "5mCNDUymtaD6",
        "outputId": "369eeb17-9245-477e-f131-30eb8f0e1666"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "100%|██████████| 29/29 [00:09<00:00,  2.98it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                               Accuracy  Balanced Accuracy  ROC AUC  F1 Score  \\\n",
            "Model                                                                           \n",
            "NearestCentroid                    0.72               0.59     0.59      0.78   \n",
            "GaussianNB                         0.87               0.58     0.58      0.87   \n",
            "QuadraticDiscriminantAnalysis      0.79               0.58     0.58      0.82   \n",
            "DecisionTreeClassifier             0.84               0.55     0.55      0.85   \n",
            "PassiveAggressiveClassifier        0.84               0.55     0.55      0.85   \n",
            "BernoulliNB                        0.91               0.55     0.55      0.89   \n",
            "LinearDiscriminantAnalysis         0.91               0.55     0.55      0.89   \n",
            "BaggingClassifier                  0.91               0.54     0.54      0.89   \n",
            "Perceptron                         0.86               0.54     0.54      0.86   \n",
            "XGBClassifier                      0.92               0.52     0.52      0.89   \n",
            "LabelPropagation                   0.87               0.52     0.52      0.86   \n",
            "LabelSpreading                     0.87               0.52     0.52      0.86   \n",
            "AdaBoostClassifier                 0.91               0.52     0.52      0.88   \n",
            "ExtraTreeClassifier                0.86               0.51     0.51      0.86   \n",
            "RandomForestClassifier             0.92               0.51     0.51      0.88   \n",
            "LogisticRegression                 0.92               0.51     0.51      0.88   \n",
            "KNeighborsClassifier               0.91               0.51     0.51      0.88   \n",
            "RidgeClassifier                    0.92               0.51     0.51      0.88   \n",
            "LinearSVC                          0.92               0.51     0.51      0.88   \n",
            "CalibratedClassifierCV             0.92               0.51     0.51      0.88   \n",
            "LGBMClassifier                     0.92               0.51     0.51      0.88   \n",
            "ExtraTreesClassifier               0.92               0.51     0.51      0.88   \n",
            "SGDClassifier                      0.91               0.50     0.50      0.88   \n",
            "RidgeClassifierCV                  0.92               0.50     0.50      0.88   \n",
            "SVC                                0.92               0.50     0.50      0.88   \n",
            "DummyClassifier                    0.84               0.49     0.49      0.84   \n",
            "\n",
            "                               Time Taken  \n",
            "Model                                      \n",
            "NearestCentroid                      0.04  \n",
            "GaussianNB                           0.03  \n",
            "QuadraticDiscriminantAnalysis        0.05  \n",
            "DecisionTreeClassifier               0.18  \n",
            "PassiveAggressiveClassifier          0.05  \n",
            "BernoulliNB                          0.06  \n",
            "LinearDiscriminantAnalysis           0.09  \n",
            "BaggingClassifier                    1.03  \n",
            "Perceptron                           0.05  \n",
            "XGBClassifier                        0.92  \n",
            "LabelPropagation                     0.63  \n",
            "LabelSpreading                       0.76  \n",
            "AdaBoostClassifier                   0.84  \n",
            "ExtraTreeClassifier                  0.03  \n",
            "RandomForestClassifier               1.19  \n",
            "LogisticRegression                   0.07  \n",
            "KNeighborsClassifier                 0.15  \n",
            "RidgeClassifier                      0.05  \n",
            "LinearSVC                            0.37  \n",
            "CalibratedClassifierCV               1.58  \n",
            "LGBMClassifier                       0.43  \n",
            "ExtraTreesClassifier                 0.40  \n",
            "SGDClassifier                        0.07  \n",
            "RidgeClassifierCV                    0.06  \n",
            "SVC                                  0.44  \n",
            "DummyClassifier                      0.03  \n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "\n",
        "\n",
        "X = train32.drop(['covid19'], axis=1)\n",
        "y= train32['covid19']\n",
        "\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.2,random_state =123)\n",
        "\n",
        "clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)\n",
        "models,predictions = clf.fit(X_train, X_test, y_train, y_test)\n",
        "\n",
        "print(models)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "fzM3-81-uly2",
        "outputId": "85a23bff-b6a0-4691-d4f1-048b35e4ae18"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "100%|██████████| 29/29 [00:13<00:00,  2.08it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                               Accuracy  Balanced Accuracy  ROC AUC  F1 Score  \\\n",
            "Model                                                                           \n",
            "NearestCentroid                    0.70               0.60     0.60      0.76   \n",
            "GaussianNB                         0.85               0.57     0.57      0.86   \n",
            "BernoulliNB                        0.91               0.56     0.56      0.89   \n",
            "QuadraticDiscriminantAnalysis      0.89               0.55     0.55      0.88   \n",
            "DecisionTreeClassifier             0.86               0.55     0.55      0.86   \n",
            "LinearDiscriminantAnalysis         0.91               0.55     0.55      0.89   \n",
            "SGDClassifier                      0.91               0.53     0.53      0.88   \n",
            "AdaBoostClassifier                 0.92               0.53     0.53      0.89   \n",
            "LogisticRegression                 0.92               0.52     0.52      0.88   \n",
            "XGBClassifier                      0.92               0.52     0.52      0.88   \n",
            "PassiveAggressiveClassifier        0.81               0.52     0.52      0.83   \n",
            "LabelPropagation                   0.88               0.52     0.52      0.87   \n",
            "LabelSpreading                     0.88               0.52     0.52      0.87   \n",
            "ExtraTreeClassifier                0.85               0.51     0.51      0.85   \n",
            "BaggingClassifier                  0.90               0.51     0.51      0.88   \n",
            "KNeighborsClassifier               0.91               0.51     0.51      0.88   \n",
            "RidgeClassifierCV                  0.92               0.51     0.51      0.88   \n",
            "RidgeClassifier                    0.92               0.51     0.51      0.88   \n",
            "LinearSVC                          0.92               0.51     0.51      0.88   \n",
            "ExtraTreesClassifier               0.92               0.51     0.51      0.88   \n",
            "CalibratedClassifierCV             0.92               0.51     0.51      0.88   \n",
            "LGBMClassifier                     0.92               0.51     0.51      0.88   \n",
            "Perceptron                         0.88               0.50     0.50      0.86   \n",
            "RandomForestClassifier             0.92               0.50     0.50      0.88   \n",
            "SVC                                0.92               0.50     0.50      0.88   \n",
            "DummyClassifier                    0.84               0.49     0.49      0.84   \n",
            "\n",
            "                               Time Taken  \n",
            "Model                                      \n",
            "NearestCentroid                      0.05  \n",
            "GaussianNB                           0.04  \n",
            "BernoulliNB                          0.04  \n",
            "QuadraticDiscriminantAnalysis        0.10  \n",
            "DecisionTreeClassifier               0.27  \n",
            "LinearDiscriminantAnalysis           0.18  \n",
            "SGDClassifier                        0.12  \n",
            "AdaBoostClassifier                   0.96  \n",
            "LogisticRegression                   0.21  \n",
            "XGBClassifier                        1.32  \n",
            "PassiveAggressiveClassifier          0.06  \n",
            "LabelPropagation                     0.81  \n",
            "LabelSpreading                       1.08  \n",
            "ExtraTreeClassifier                  0.03  \n",
            "BaggingClassifier                    1.44  \n",
            "KNeighborsClassifier                 0.32  \n",
            "RidgeClassifierCV                    0.08  \n",
            "RidgeClassifier                      0.04  \n",
            "LinearSVC                            0.72  \n",
            "ExtraTreesClassifier                 0.49  \n",
            "CalibratedClassifierCV               1.89  \n",
            "LGBMClassifier                       0.76  \n",
            "Perceptron                           0.05  \n",
            "RandomForestClassifier               2.07  \n",
            "SVC                                  0.60  \n",
            "DummyClassifier                      0.04  \n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "\n",
        "\n",
        "X = train64.drop(['covid19'], axis=1)\n",
        "y= train64['covid19']\n",
        "\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.2,random_state =123)\n",
        "\n",
        "clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)\n",
        "models,predictions = clf.fit(X_train, X_test, y_train, y_test)\n",
        "\n",
        "print(models)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "FuOeofofvFaZ",
        "outputId": "bc0593a2-19f4-4041-ffc5-f9999c8c1314"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "100%|██████████| 29/29 [00:22<00:00,  1.31it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "                               Accuracy  Balanced Accuracy  ROC AUC  F1 Score  \\\n",
            "Model                                                                           \n",
            "NearestCentroid                    0.71               0.61     0.61      0.77   \n",
            "GaussianNB                         0.86               0.59     0.59      0.86   \n",
            "LinearDiscriminantAnalysis         0.91               0.55     0.55      0.89   \n",
            "QuadraticDiscriminantAnalysis      0.84               0.54     0.54      0.85   \n",
            "SGDClassifier                      0.90               0.54     0.54      0.88   \n",
            "DecisionTreeClassifier             0.84               0.54     0.54      0.85   \n",
            "Perceptron                         0.90               0.53     0.53      0.88   \n",
            "BernoulliNB                        0.91               0.53     0.53      0.88   \n",
            "AdaBoostClassifier                 0.91               0.53     0.53      0.88   \n",
            "LabelPropagation                   0.91               0.52     0.52      0.88   \n",
            "LabelSpreading                     0.91               0.52     0.52      0.88   \n",
            "LogisticRegression                 0.92               0.51     0.51      0.88   \n",
            "ExtraTreeClassifier                0.84               0.51     0.51      0.84   \n",
            "ExtraTreesClassifier               0.92               0.51     0.51      0.88   \n",
            "LGBMClassifier                     0.92               0.51     0.51      0.88   \n",
            "CalibratedClassifierCV             0.92               0.51     0.51      0.88   \n",
            "XGBClassifier                      0.91               0.51     0.51      0.88   \n",
            "PassiveAggressiveClassifier        0.85               0.51     0.51      0.85   \n",
            "KNeighborsClassifier               0.91               0.51     0.51      0.88   \n",
            "BaggingClassifier                  0.90               0.50     0.50      0.87   \n",
            "RandomForestClassifier             0.92               0.50     0.50      0.88   \n",
            "RidgeClassifier                    0.92               0.50     0.50      0.88   \n",
            "RidgeClassifierCV                  0.92               0.50     0.50      0.88   \n",
            "SVC                                0.92               0.50     0.50      0.88   \n",
            "LinearSVC                          0.91               0.50     0.50      0.88   \n",
            "DummyClassifier                    0.84               0.49     0.49      0.84   \n",
            "\n",
            "                               Time Taken  \n",
            "Model                                      \n",
            "NearestCentroid                      0.06  \n",
            "GaussianNB                           0.05  \n",
            "LinearDiscriminantAnalysis           0.14  \n",
            "QuadraticDiscriminantAnalysis        0.08  \n",
            "SGDClassifier                        0.14  \n",
            "DecisionTreeClassifier               0.51  \n",
            "Perceptron                           0.06  \n",
            "BernoulliNB                          0.06  \n",
            "AdaBoostClassifier                   1.53  \n",
            "LabelPropagation                     0.51  \n",
            "LabelSpreading                       0.67  \n",
            "LogisticRegression                   0.16  \n",
            "ExtraTreeClassifier                  0.04  \n",
            "ExtraTreesClassifier                 0.56  \n",
            "LGBMClassifier                       4.74  \n",
            "CalibratedClassifierCV               2.68  \n",
            "XGBClassifier                        2.50  \n",
            "PassiveAggressiveClassifier          0.07  \n",
            "KNeighborsClassifier                 0.40  \n",
            "BaggingClassifier                    2.65  \n",
            "RandomForestClassifier               2.53  \n",
            "RidgeClassifier                      0.05  \n",
            "RidgeClassifierCV                    0.09  \n",
            "SVC                                  0.96  \n",
            "LinearSVC                            0.76  \n",
            "DummyClassifier                      0.04  \n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        ""
      ],
      "metadata": {
        "id": "tknziOPyvPvX"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}