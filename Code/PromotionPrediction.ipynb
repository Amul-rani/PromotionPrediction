{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "79acf6ae-6817-4243-8226-73ab6429bcb4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\"\\n            Analyzing Promotion Prediction Data\\nObjective\\n    The goal of this hackathon is to use data analysis and machine learning to predict promotion.\\nThis means you'll determine identifying the right people for promotion, based on past data\\n\""
      ]
     },
     "execution_count": 1,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "'''\n",
    "            Analyzing Promotion Prediction Data\n",
    "Objective\n",
    "    The goal of this hackathon is to use data analysis and machine learning to predict promotion.\n",
    "This means you'll determine identifying the right people for promotion, based on past data\n",
    "'''"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "id": "8052d3d3-bfc9-42c8-ac94-2c740fa62798",
   "metadata": {},
   "outputs": [],
   "source": [
    "# import libraries\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "import sklearn\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.impute import SimpleImputer\n",
    "from sklearn.preprocessing import OneHotEncoder, StandardScaler\n",
    "from sklearn.compose import ColumnTransformer\n",
    "\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc\n",
    "from sklearn.model_selection import train_test_split, GridSearchCV\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "\n",
    "import joblib\n",
    "from imblearn.over_sampling import RandomOverSampler\n",
    "import imblearn\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "1f8b676d-1baa-4edd-988f-2cd937fdfd49",
   "metadata": {},
   "outputs": [],
   "source": [
    "# read the test and train data\n",
    "train = pd.read_csv('../Data/train_LZdllcl.csv')\n",
    "test  = pd.read_csv('../Data/test_2umaH9m.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "187d139c-c787-443e-91ad-b37e940544ee",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((54808, 14), (23490, 13))"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# total no.of rows and columns\n",
    "train.shape, test.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "22b0d9af-c16d-4b9f-83be-8ede5c7d212a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "employee_id               int64\n",
       "department               object\n",
       "region                   object\n",
       "education                object\n",
       "gender                   object\n",
       "recruitment_channel      object\n",
       "no_of_trainings           int64\n",
       "age                       int64\n",
       "previous_year_rating    float64\n",
       "length_of_service         int64\n",
       "KPIs_met >80%             int64\n",
       "awards_won?               int64\n",
       "avg_training_score        int64\n",
       "is_promoted               int64\n",
       "dtype: object"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# to view the dataypes\n",
    "train.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "c24162c3-13d7-4c8b-a424-b33035b23bd6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 54808 entries, 0 to 54807\n",
      "Data columns (total 14 columns):\n",
      " #   Column                Non-Null Count  Dtype  \n",
      "---  ------                --------------  -----  \n",
      " 0   employee_id           54808 non-null  int64  \n",
      " 1   department            54808 non-null  object \n",
      " 2   region                54808 non-null  object \n",
      " 3   education             52399 non-null  object \n",
      " 4   gender                54808 non-null  object \n",
      " 5   recruitment_channel   54808 non-null  object \n",
      " 6   no_of_trainings       54808 non-null  int64  \n",
      " 7   age                   54808 non-null  int64  \n",
      " 8   previous_year_rating  50684 non-null  float64\n",
      " 9   length_of_service     54808 non-null  int64  \n",
      " 10  KPIs_met >80%         54808 non-null  int64  \n",
      " 11  awards_won?           54808 non-null  int64  \n",
      " 12  avg_training_score    54808 non-null  int64  \n",
      " 13  is_promoted           54808 non-null  int64  \n",
      "dtypes: float64(1), int64(8), object(5)\n",
      "memory usage: 5.9+ MB\n"
     ]
    }
   ],
   "source": [
    "# get all details of the dataset\n",
    "train.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "90f6384d-e970-4e6b-97d9-845ed083da65",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>employee_id</th>\n",
       "      <th>department</th>\n",
       "      <th>region</th>\n",
       "      <th>education</th>\n",
       "      <th>gender</th>\n",
       "      <th>recruitment_channel</th>\n",
       "      <th>no_of_trainings</th>\n",
       "      <th>age</th>\n",
       "      <th>previous_year_rating</th>\n",
       "      <th>length_of_service</th>\n",
       "      <th>KPIs_met &gt;80%</th>\n",
       "      <th>awards_won?</th>\n",
       "      <th>avg_training_score</th>\n",
       "      <th>is_promoted</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>65438</td>\n",
       "      <td>Sales &amp; Marketing</td>\n",
       "      <td>region_7</td>\n",
       "      <td>Master's &amp; above</td>\n",
       "      <td>f</td>\n",
       "      <td>sourcing</td>\n",
       "      <td>1</td>\n",
       "      <td>35</td>\n",
       "      <td>5.0</td>\n",
       "      <td>8</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>49</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>65141</td>\n",
       "      <td>Operations</td>\n",
       "      <td>region_22</td>\n",
       "      <td>Bachelor's</td>\n",
       "      <td>m</td>\n",
       "      <td>other</td>\n",
       "      <td>1</td>\n",
       "      <td>30</td>\n",
       "      <td>5.0</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>60</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   employee_id         department     region         education gender  \\\n",
       "0        65438  Sales & Marketing   region_7  Master's & above      f   \n",
       "1        65141         Operations  region_22        Bachelor's      m   \n",
       "\n",
       "  recruitment_channel  no_of_trainings  age  previous_year_rating  \\\n",
       "0            sourcing                1   35                   5.0   \n",
       "1               other                1   30                   5.0   \n",
       "\n",
       "   length_of_service  KPIs_met >80%  awards_won?  avg_training_score  \\\n",
       "0                  8              1            0                  49   \n",
       "1                  4              0            0                  60   \n",
       "\n",
       "   is_promoted  \n",
       "0            0  \n",
       "1            0  "
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# to view the first two rows\n",
    "train.head(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "ac3099aa-e333-48fb-8aed-73d16939ab7e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "employee_id                0\n",
       "department                 0\n",
       "region                     0\n",
       "education               2409\n",
       "gender                     0\n",
       "recruitment_channel        0\n",
       "no_of_trainings            0\n",
       "age                        0\n",
       "previous_year_rating    4124\n",
       "length_of_service          0\n",
       "KPIs_met >80%              0\n",
       "awards_won?                0\n",
       "avg_training_score         0\n",
       "is_promoted                0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# checking missing data\n",
    "train.isna().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "5d67e62e-9377-403a-8638-c3ebe7debfa1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "employee_id             0.000000\n",
       "department              0.000000\n",
       "region                  0.000000\n",
       "education               4.395344\n",
       "gender                  0.000000\n",
       "recruitment_channel     0.000000\n",
       "no_of_trainings         0.000000\n",
       "age                     0.000000\n",
       "previous_year_rating    7.524449\n",
       "length_of_service       0.000000\n",
       "KPIs_met >80%           0.000000\n",
       "awards_won?             0.000000\n",
       "avg_training_score      0.000000\n",
       "is_promoted             0.000000\n",
       "dtype: float64"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.isna().sum()/train.shape[0]*100"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "c70a8413-a47b-4d81-8440-68904a5927c0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "employee_id             0.000000\n",
       "department              0.000000\n",
       "region                  0.000000\n",
       "gender                  0.000000\n",
       "recruitment_channel     0.000000\n",
       "no_of_trainings         0.000000\n",
       "age                     0.000000\n",
       "length_of_service       0.000000\n",
       "KPIs_met >80%           0.000000\n",
       "awards_won?             0.000000\n",
       "avg_training_score      0.000000\n",
       "is_promoted             0.000000\n",
       "education               4.395344\n",
       "previous_year_rating    7.524449\n",
       "dtype: float64"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# percentage of missing data in each column\n",
    "(train.isna().sum()/train.shape[0]*100).sort_values()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "04ac4794-2b3d-4c02-ba71-693baf59b137",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# to check duplicates\n",
    "train.duplicated().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "8b0b0112-28b9-42bb-a742-d9f503d05e2c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "is_promoted\n",
       "0    50140\n",
       "1     4668\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.is_promoted.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "2c130921-c390-45d2-9941-9af797affeca",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "is_promoted\n",
       "0    91.482995\n",
       "1     8.517005\n",
       "Name: proportion, dtype: float64"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train['is_promoted'].value_counts(normalize=True)*100"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "6fed0c49-038c-4b86-b04c-5d303b543d5b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 91.483% of the employees not got the promotion\n",
    "# only 8.517 employees got the promotion"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "e7bc2241-96b6-4185-8ae3-cc44e7b85df8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='is_promoted'>"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjoAAAGrCAYAAADJmj27AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy81sbWrAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAoAUlEQVR4nO3df1DU953H8dcKskGEb0Bk151QNS3HQTA5Q3qIttVWBR2ROr0bM0dvG+csmmIkVDiN12tjvQ4majTtcU1Mehfzw4TrjPGaqcrBXe9sOMAf5GiCUePNacTIionropRbCH7vj47fyYIxokbk4/MxszPd7/f9/e5nSQnPfNldXLZt2wIAADDQiKFeAAAAwOeF0AEAAMYidAAAgLEIHQAAYCxCBwAAGIvQAQAAxiJ0AACAsaKHegFD6eLFizp16pTi4+PlcrmGejkAAOAq2Lat8+fPy+fzacSIK1+zua1D59SpU0pNTR3qZQAAgGvQ1tamu+6664ozt3XoxMfHS/rDFyohIWGIVwMAAK5GZ2enUlNTnZ/jV3Jbh86lX1clJCQQOgAADDNX87ITXowMAACMRegAAABjEToAAMBYhA4AADAWoQMAAIxF6AAAAGMROgAAwFiEDgAAMBahAwAAjEXoAAAAYxE6AADAWIQOAAAw1qBCZ82aNXK5XBE3r9fr7LdtW2vWrJHP51NsbKxmzJihgwcPRpwjHA5r+fLlSk5OVlxcnAoLC3Xy5MmImWAwKL/fL8uyZFmW/H6/zp07FzFz4sQJzZ8/X3FxcUpOTlZpaal6enoG+fQBAIDJBn1F55577lF7e7tze+edd5x969ev16ZNm1RVVaX9+/fL6/Vq9uzZOn/+vDNTVlamHTt2qLq6WvX19bpw4YIKCgrU19fnzBQVFamlpUU1NTWqqalRS0uL/H6/s7+vr0/z5s1TV1eX6uvrVV1dre3bt6u8vPxavw4AAMBE9iA8/vjj9n333XfZfRcvXrS9Xq/9xBNPONv+7//+z7Ysy3722Wdt27btc+fO2SNHjrSrq6udmQ8++MAeMWKEXVNTY9u2bb/77ru2JLupqcmZaWxstCXZhw8ftm3btnft2mWPGDHC/uCDD5yZ1157zXa73XYoFLrq5xMKhWxJgzoGAAAMrcH8/I4ebBgdPXpUPp9PbrdbOTk5qqys1N13361jx44pEAgoLy/PmXW73Zo+fboaGhq0dOlSNTc3q7e3N2LG5/MpKytLDQ0Nys/PV2NjoyzLUk5OjjMzZcoUWZalhoYGpaenq7GxUVlZWfL5fM5Mfn6+wuGwmpub9fWvf/2yaw+HwwqHw879zs7OwT59Y0x4bOdQLwE30fEn5g31EgBgSAzqV1c5OTl66aWX9K//+q96/vnnFQgENHXqVH300UcKBAKSJI/HE3GMx+Nx9gUCAcXExCgxMfGKMykpKQMeOyUlJWKm/+MkJiYqJibGmbmcdevWOa/7sSxLqampg3n6AABgmBlU6MydO1d/9md/pkmTJmnWrFnaufMPVwVefPFFZ8blckUcY9v2gG399Z+53Py1zPS3evVqhUIh59bW1nbFdQEAgOHtut5eHhcXp0mTJuno0aPOu6/6X1Hp6Ohwrr54vV719PQoGAxeceb06dMDHuvMmTMRM/0fJxgMqre3d8CVnk9yu91KSEiIuAEAAHNdV+iEw2EdOnRI48aN08SJE+X1elVXV+fs7+np0Z49ezR16lRJUnZ2tkaOHBkx097ertbWVmcmNzdXoVBI+/btc2b27t2rUCgUMdPa2qr29nZnpra2Vm63W9nZ2dfzlAAAgEEG9WLkiooKzZ8/X1/4whfU0dGhn/zkJ+rs7NRDDz0kl8ulsrIyVVZWKi0tTWlpaaqsrNSoUaNUVFQkSbIsS4sXL1Z5ebnGjBmjpKQkVVRUOL8Kk6SMjAzNmTNHxcXF2rJliyRpyZIlKigoUHp6uiQpLy9PmZmZ8vv92rBhg86ePauKigoVFxdzlQYAADgGFTonT57UX/zFX+jDDz/U2LFjNWXKFDU1NWn8+PGSpJUrV6q7u1slJSUKBoPKyclRbW2t4uPjnXNs3rxZ0dHRWrhwobq7uzVz5kxt3bpVUVFRzsy2bdtUWlrqvDursLBQVVVVzv6oqCjt3LlTJSUlmjZtmmJjY1VUVKSNGzde1xcDAACYxWXbtj3UixgqnZ2dsixLoVDotrsSxNvLby+8vRyASQbz85u/dQUAAIxF6AAAAGMROgAAwFiEDgAAMBahAwAAjEXoAAAAYxE6AADAWIQOAAAwFqEDAACMRegAAABjEToAAMBYhA4AADAWoQMAAIxF6AAAAGMROgAAwFiEDgAAMBahAwAAjEXoAAAAYxE6AADAWIQOAAAwFqEDAACMRegAAABjEToAAMBYhA4AADAWoQMAAIxF6AAAAGMROgAAwFiEDgAAMBahAwAAjEXoAAAAYxE6AADAWIQOAAAwFqEDAACMRegAAABjEToAAMBYhA4AADAWoQMAAIxF6AAAAGMROgAAwFiEDgAAMBahAwAAjEXoAAAAYxE6AADAWIQOAAAwFqEDAACMRegAAABjEToAAMBYhA4AADAWoQMAAIxF6AAAAGMROgAAwFiEDgAAMBahAwAAjEXoAAAAYxE6AADAWIQOAAAwFqEDAACMRegAAABjEToAAMBY1xU669atk8vlUllZmbPNtm2tWbNGPp9PsbGxmjFjhg4ePBhxXDgc1vLly5WcnKy4uDgVFhbq5MmTETPBYFB+v1+WZcmyLPn9fp07dy5i5sSJE5o/f77i4uKUnJys0tJS9fT0XM9TAgAABrnm0Nm/f7+ee+453XvvvRHb169fr02bNqmqqkr79++X1+vV7Nmzdf78eWemrKxMO3bsUHV1terr63XhwgUVFBSor6/PmSkqKlJLS4tqampUU1OjlpYW+f1+Z39fX5/mzZunrq4u1dfXq7q6Wtu3b1d5efm1PiUAAGCYawqdCxcu6Nvf/raef/55JSYmOttt29bTTz+tH/zgB/rWt76lrKwsvfjii/r973+vV199VZIUCoX0j//4j3rqqac0a9YsTZ48Wa+88oreeecd/du//Zsk6dChQ6qpqdEvfvEL5ebmKjc3V88//7x+/etf68iRI5Kk2tpavfvuu3rllVc0efJkzZo1S0899ZSef/55dXZ2Xu/XBQAAGOCaQmfZsmWaN2+eZs2aFbH92LFjCgQCysvLc7a53W5Nnz5dDQ0NkqTm5mb19vZGzPh8PmVlZTkzjY2NsixLOTk5zsyUKVNkWVbETFZWlnw+nzOTn5+vcDis5ubmy647HA6rs7Mz4gYAAMwVPdgDqqur9dZbb2n//v0D9gUCAUmSx+OJ2O7xePT+++87MzExMRFXgi7NXDo+EAgoJSVlwPlTUlIiZvo/TmJiomJiYpyZ/tatW6cf//jHV/M0AQCAAQZ1RaetrU2PPvqoXnnlFd1xxx2fOudyuSLu27Y9YFt//WcuN38tM5+0evVqhUIh59bW1nbFNQEAgOFtUKHT3Nysjo4OZWdnKzo6WtHR0dqzZ49+9rOfKTo62rnC0v+KSkdHh7PP6/Wqp6dHwWDwijOnT58e8PhnzpyJmOn/OMFgUL29vQOu9FzidruVkJAQcQMAAOYaVOjMnDlT77zzjlpaWpzbAw88oG9/+9tqaWnR3XffLa/Xq7q6OueYnp4e7dmzR1OnTpUkZWdna+TIkREz7e3tam1tdWZyc3MVCoW0b98+Z2bv3r0KhUIRM62trWpvb3dmamtr5Xa7lZ2dfQ1fCgAAYJpBvUYnPj5eWVlZEdvi4uI0ZswYZ3tZWZkqKyuVlpamtLQ0VVZWatSoUSoqKpIkWZalxYsXq7y8XGPGjFFSUpIqKio0adIk58XNGRkZmjNnjoqLi7VlyxZJ0pIlS1RQUKD09HRJUl5enjIzM+X3+7VhwwadPXtWFRUVKi4u5koNAACQdA0vRv4sK1euVHd3t0pKShQMBpWTk6Pa2lrFx8c7M5s3b1Z0dLQWLlyo7u5uzZw5U1u3blVUVJQzs23bNpWWljrvziosLFRVVZWzPyoqSjt37lRJSYmmTZum2NhYFRUVaePGjTf6KQEAgGHKZdu2PdSLGCqdnZ2yLEuhUOi2uwo04bGdQ70E3ETHn5g31EsAgBtmMD+/+VtXAADAWIQOAAAwFqEDAACMRegAAABjEToAAMBYhA4AADAWoQMAAIxF6AAAAGMROgAAwFiEDgAAMBahAwAAjEXoAAAAYxE6AADAWIQOAAAwFqEDAACMRegAAABjEToAAMBYhA4AADAWoQMAAIxF6AAAAGMROgAAwFiEDgAAMBahAwAAjEXoAAAAYxE6AADAWIQOAAAwFqEDAACMRegAAABjEToAAMBYhA4AADAWoQMAAIxF6AAAAGMROgAAwFiEDgAAMBahAwAAjEXoAAAAYxE6AADAWIQOAAAwFqEDAACMRegAAABjEToAAMBYhA4AADAWoQMAAIxF6AAAAGMROgAAwFiEDgAAMBahAwAAjEXoAAAAYxE6AADAWIQOAAAwFqEDAACMRegAAABjEToAAMBYhA4AADAWoQMAAIxF6AAAAGMROgAAwFiEDgAAMBahAwAAjDWo0HnmmWd07733KiEhQQkJCcrNzdXu3bud/bZta82aNfL5fIqNjdWMGTN08ODBiHOEw2EtX75cycnJiouLU2FhoU6ePBkxEwwG5ff7ZVmWLMuS3+/XuXPnImZOnDih+fPnKy4uTsnJySotLVVPT88gnz4AADDZoELnrrvu0hNPPKEDBw7owIED+sY3vqFvfvObTsysX79emzZtUlVVlfbv3y+v16vZs2fr/PnzzjnKysq0Y8cOVVdXq76+XhcuXFBBQYH6+vqcmaKiIrW0tKimpkY1NTVqaWmR3+939vf19WnevHnq6upSfX29qqurtX37dpWXl1/v1wMAABjEZdu2fT0nSEpK0oYNG/RXf/VX8vl8Kisr06pVqyT94eqNx+PRk08+qaVLlyoUCmns2LF6+eWX9eCDD0qSTp06pdTUVO3atUv5+fk6dOiQMjMz1dTUpJycHElSU1OTcnNzdfjwYaWnp2v37t0qKChQW1ubfD6fJKm6ulqLFi1SR0eHEhISrmrtnZ2dsixLoVDoqo8xxYTHdg71EnATHX9i3lAvAQBumMH8/L7m1+j09fWpurpaXV1dys3N1bFjxxQIBJSXl+fMuN1uTZ8+XQ0NDZKk5uZm9fb2Rsz4fD5lZWU5M42NjbIsy4kcSZoyZYosy4qYycrKciJHkvLz8xUOh9Xc3Pypaw6Hw+rs7Iy4AQAAcw06dN555x2NHj1abrdbDz/8sHbs2KHMzEwFAgFJksfjiZj3eDzOvkAgoJiYGCUmJl5xJiUlZcDjpqSkRMz0f5zExETFxMQ4M5ezbt0653U/lmUpNTV1kM8eAAAMJ4MOnfT0dLW0tKipqUnf+9739NBDD+ndd9919rtcroh527YHbOuv/8zl5q9lpr/Vq1crFAo5t7a2tiuuCwAADG+DDp2YmBh96Utf0gMPPKB169bpvvvu009/+lN5vV5JGnBFpaOjw7n64vV61dPTo2AweMWZ06dPD3jcM2fORMz0f5xgMKje3t4BV3o+ye12O+8Yu3QDAADmuu7P0bFtW+FwWBMnTpTX61VdXZ2zr6enR3v27NHUqVMlSdnZ2Ro5cmTETHt7u1pbW52Z3NxchUIh7du3z5nZu3evQqFQxExra6va29udmdraWrndbmVnZ1/vUwIAAIaIHszw3/zN32ju3LlKTU3V+fPnVV1drf/8z/9UTU2NXC6XysrKVFlZqbS0NKWlpamyslKjRo1SUVGRJMmyLC1evFjl5eUaM2aMkpKSVFFRoUmTJmnWrFmSpIyMDM2ZM0fFxcXasmWLJGnJkiUqKChQenq6JCkvL0+ZmZny+/3asGGDzp49q4qKChUXF3OVBgAAOAYVOqdPn5bf71d7e7ssy9K9996rmpoazZ49W5K0cuVKdXd3q6SkRMFgUDk5OaqtrVV8fLxzjs2bNys6OloLFy5Ud3e3Zs6cqa1btyoqKsqZ2bZtm0pLS513ZxUWFqqqqsrZHxUVpZ07d6qkpETTpk1TbGysioqKtHHjxuv6YgAAALNc9+foDGd8jg5uF3yODgCT3JTP0QEAALjVEToAAMBYhA4AADAWoQMAAIxF6AAAAGMROgAAwFiEDgAAMBahAwAAjEXoAAAAYxE6AADAWIQOAAAwFqEDAACMRegAAABjEToAAMBYhA4AADAWoQMAAIxF6AAAAGMROgAAwFiEDgAAMBahAwAAjEXoAAAAYxE6AADAWIQOAAAwFqEDAACMRegAAABjEToAAMBYhA4AADAWoQMAAIxF6AAAAGMROgAAwFiEDgAAMBahAwAAjEXoAAAAYxE6AADAWIQOAAAwFqEDAACMRegAAABjEToAAMBYhA4AADAWoQMAAIxF6AAAAGMROgAAwFiEDgAAMBahAwAAjEXoAAAAYxE6AADAWIQOAAAwFqEDAACMRegAAABjEToAAMBYhA4AADAWoQMAAIxF6AAAAGMROgAAwFiEDgAAMBahAwAAjEXoAAAAYxE6AADAWIQOAAAwFqEDAACMNajQWbdunb785S8rPj5eKSkpWrBggY4cORIxY9u21qxZI5/Pp9jYWM2YMUMHDx6MmAmHw1q+fLmSk5MVFxenwsJCnTx5MmImGAzK7/fLsixZliW/369z585FzJw4cULz589XXFyckpOTVVpaqp6ensE8JQAAYLBBhc6ePXu0bNkyNTU1qa6uTh9//LHy8vLU1dXlzKxfv16bNm1SVVWV9u/fL6/Xq9mzZ+v8+fPOTFlZmXbs2KHq6mrV19frwoULKigoUF9fnzNTVFSklpYW1dTUqKamRi0tLfL7/c7+vr4+zZs3T11dXaqvr1d1dbW2b9+u8vLy6/l6AAAAg7hs27av9eAzZ84oJSVFe/bs0de+9jXZti2fz6eysjKtWrVK0h+u3ng8Hj355JNaunSpQqGQxo4dq5dfflkPPvigJOnUqVNKTU3Vrl27lJ+fr0OHDikzM1NNTU3KycmRJDU1NSk3N1eHDx9Wenq6du/erYKCArW1tcnn80mSqqurtWjRInV0dCghIWHAesPhsMLhsHO/s7NTqampCoVCl5032YTHdg71EnATHX9i3lAvAQBumM7OTlmWdVU/v6/rNTqhUEiSlJSUJEk6duyYAoGA8vLynBm3263p06eroaFBktTc3Kze3t6IGZ/Pp6ysLGemsbFRlmU5kSNJU6ZMkWVZETNZWVlO5EhSfn6+wuGwmpubL7vedevWOb8KsyxLqamp1/P0AQDALe6aQ8e2ba1YsUJf+cpXlJWVJUkKBAKSJI/HEzHr8XicfYFAQDExMUpMTLziTEpKyoDHTElJiZjp/ziJiYmKiYlxZvpbvXq1QqGQc2traxvs0wYAAMNI9LUe+Mgjj+jtt99WfX39gH0ulyvivm3bA7b113/mcvPXMvNJbrdbbrf7iusAAADmuKYrOsuXL9cbb7yh//iP/9Bdd93lbPd6vZI04IpKR0eHc/XF6/Wqp6dHwWDwijOnT58e8LhnzpyJmOn/OMFgUL29vQOu9AAAgNvToELHtm098sgjev311/Wb3/xGEydOjNg/ceJEeb1e1dXVOdt6enq0Z88eTZ06VZKUnZ2tkSNHRsy0t7ertbXVmcnNzVUoFNK+ffucmb179yoUCkXMtLa2qr293Zmpra2V2+1Wdnb2YJ4WAAAw1KB+dbVs2TK9+uqr+tWvfqX4+HjnioplWYqNjZXL5VJZWZkqKyuVlpamtLQ0VVZWatSoUSoqKnJmFy9erPLyco0ZM0ZJSUmqqKjQpEmTNGvWLElSRkaG5syZo+LiYm3ZskWStGTJEhUUFCg9PV2SlJeXp8zMTPn9fm3YsEFnz55VRUWFiouLb7t3UAEAgMsbVOg888wzkqQZM2ZEbH/hhRe0aNEiSdLKlSvV3d2tkpISBYNB5eTkqLa2VvHx8c785s2bFR0drYULF6q7u1szZ87U1q1bFRUV5cxs27ZNpaWlzruzCgsLVVVV5eyPiorSzp07VVJSomnTpik2NlZFRUXauHHjoL4AAADAXNf1OTrD3WDeh28aPkfn9sLn6AAwyU37HB0AAIBbGaEDAACMRegAAABjEToAAMBYhA4AADAWoQMAAIxF6AAAAGMROgAAwFiEDgAAMBahAwAAjEXoAAAAYxE6AADAWIQOAAAwFqEDAACMRegAAABjEToAAMBYhA4AADAWoQMAAIxF6AAAAGMROgAAwFiEDgAAMBahAwAAjEXoAAAAYxE6AADAWIQOAAAwFqEDAACMRegAAABjEToAAMBYhA4AADAWoQMAAIxF6AAAAGMROgAAwFiEDgAAMBahAwAAjEXoAAAAYxE6AADAWIQOAAAwFqEDAACMRegAAABjEToAAMBYhA4AADAWoQMAAIxF6AAAAGMROgAAwFiEDgAAMBahAwAAjEXoAAAAYxE6AADAWIQOAAAwFqEDAACMRegAAABjEToAAMBYhA4AADAWoQMAAIxF6AAAAGMROgAAwFiEDgAAMBahAwAAjEXoAAAAYw06dH77299q/vz58vl8crlc+pd/+ZeI/bZta82aNfL5fIqNjdWMGTN08ODBiJlwOKzly5crOTlZcXFxKiws1MmTJyNmgsGg/H6/LMuSZVny+/06d+5cxMyJEyc0f/58xcXFKTk5WaWlperp6RnsUwIAAIYadOh0dXXpvvvuU1VV1WX3r1+/Xps2bVJVVZX2798vr9er2bNn6/z5885MWVmZduzYoerqatXX1+vChQsqKChQX1+fM1NUVKSWlhbV1NSopqZGLS0t8vv9zv6+vj7NmzdPXV1dqq+vV3V1tbZv367y8vLBPiUAAGAol23b9jUf7HJpx44dWrBggaQ/XM3x+XwqKyvTqlWrJP3h6o3H49GTTz6ppUuXKhQKaezYsXr55Zf14IMPSpJOnTql1NRU7dq1S/n5+Tp06JAyMzPV1NSknJwcSVJTU5Nyc3N1+PBhpaena/fu3SooKFBbW5t8Pp8kqbq6WosWLVJHR4cSEhI+c/2dnZ2yLEuhUOiq5k0y4bGdQ70E3ETHn5g31EsAgBtmMD+/b+hrdI4dO6ZAIKC8vDxnm9vt1vTp09XQ0CBJam5uVm9vb8SMz+dTVlaWM9PY2CjLspzIkaQpU6bIsqyImaysLCdyJCk/P1/hcFjNzc2XXV84HFZnZ2fEDQAAmOuGhk4gEJAkeTyeiO0ej8fZFwgEFBMTo8TExCvOpKSkDDh/SkpKxEz/x0lMTFRMTIwz09+6deuc1/xYlqXU1NRreJYAAGC4+FzedeVyuSLu27Y9YFt//WcuN38tM5+0evVqhUIh59bW1nbFNQEAgOHthoaO1+uVpAFXVDo6OpyrL16vVz09PQoGg1ecOX369IDznzlzJmKm/+MEg0H19vYOuNJzidvtVkJCQsQNAACY64aGzsSJE+X1elVXV+ds6+np0Z49ezR16lRJUnZ2tkaOHBkx097ertbWVmcmNzdXoVBI+/btc2b27t2rUCgUMdPa2qr29nZnpra2Vm63W9nZ2TfyaQEAgGEqerAHXLhwQf/zP//j3D927JhaWlqUlJSkL3zhCyorK1NlZaXS0tKUlpamyspKjRo1SkVFRZIky7K0ePFilZeXa8yYMUpKSlJFRYUmTZqkWbNmSZIyMjI0Z84cFRcXa8uWLZKkJUuWqKCgQOnp6ZKkvLw8ZWZmyu/3a8OGDTp79qwqKipUXFzMlRoAACDpGkLnwIED+vrXv+7cX7FihSTpoYce0tatW7Vy5Up1d3erpKREwWBQOTk5qq2tVXx8vHPM5s2bFR0drYULF6q7u1szZ87U1q1bFRUV5cxs27ZNpaWlzruzCgsLIz67JyoqSjt37lRJSYmmTZum2NhYFRUVaePGjYP/KgAAACNd1+foDHd8jg5uF3yODgCTDNnn6AAAANxKCB0AAGAsQgcAABiL0AEAAMYidAAAgLEIHQAAYCxCBwAAGIvQAQAAxiJ0AACAsQgdAABgLEIHAAAYi9ABAADGInQAAICxCB0AAGAsQgcAABiL0AEAAMYidAAAgLEIHQAAYCxCBwAAGIvQAQAAxiJ0AACAsQgdAABgLEIHAAAYi9ABAADGInQAAICxCB0AAGAsQgcAABiL0AEAAMYidAAAgLEIHQAAYKzooV4AAODGmvDYzqFeAm6i40/MG+ol3NK4ogMAAIxF6AAAAGMROgAAwFiEDgAAMBahAwAAjEXoAAAAYxE6AADAWIQOAAAwFqEDAACMRegAAABjEToAAMBYhA4AADAWoQMAAIxF6AAAAGMROgAAwFiEDgAAMBahAwAAjEXoAAAAYxE6AADAWIQOAAAwFqEDAACMRegAAABjEToAAMBYhA4AADAWoQMAAIxF6AAAAGMROgAAwFiEDgAAMNawD52f//znmjhxou644w5lZ2frzTffHOolAQCAW8SwDp1//ud/VllZmX7wgx/ov//7v/XVr35Vc+fO1YkTJ4Z6aQAA4BYwrENn06ZNWrx4sb773e8qIyNDTz/9tFJTU/XMM88M9dIAAMAtIHqoF3Ctenp61NzcrMceeyxie15enhoaGi57TDgcVjgcdu6HQiFJUmdn5+e30FvUxfDvh3oJuIlux/+P3874/r693I7f35ees23bnzk7bEPnww8/VF9fnzweT8R2j8ejQCBw2WPWrVunH//4xwO2p6amfi5rBG4V1tNDvQIAn5fb+fv7/PnzsizrijPDNnQucblcEfdt2x6w7ZLVq1drxYoVzv2LFy/q7NmzGjNmzKceA3N0dnYqNTVVbW1tSkhIGOrlALiB+P6+vdi2rfPnz8vn833m7LANneTkZEVFRQ24etPR0THgKs8lbrdbbrc7Ytudd975eS0Rt6iEhAT+RQgYiu/v28dnXcm5ZNi+GDkmJkbZ2dmqq6uL2F5XV6epU6cO0aoAAMCtZNhe0ZGkFStWyO/364EHHlBubq6ee+45nThxQg8//PBQLw0AANwChnXoPPjgg/roo4+0du1atbe3KysrS7t27dL48eOHemm4Bbndbj3++OMDfn0JYPjj+xufxmVfzXuzAAAAhqFh+xodAACAz0LoAAAAYxE6AADAWIQOAAAwFqEDAACMNazfXg5cycmTJ/XMM8+ooaFBgUBALpdLHo9HU6dO1cMPP8zfOAOA2wBvL4eR6uvrNXfuXKWmpiovL08ej0e2baujo0N1dXVqa2vT7t27NW3atKFeKoDPQVtbmx5//HH90z/901AvBUOM0IGRvvzlL+srX/mKNm/efNn93//+91VfX6/9+/ff5JUBuBl+97vf6f7771dfX99QLwVDjNCBkWJjY9XS0qL09PTL7j98+LAmT56s7u7um7wyADfCG2+8ccX9//u//6vy8nJCB7xGB2YaN26cGhoaPjV0GhsbNW7cuJu8KgA3yoIFC+RyuXSl/1Z3uVw3cUW4VRE6MFJFRYUefvhhNTc3a/bs2fJ4PHK5XAoEAqqrq9MvfvELPf3000O9TADXaNy4cfqHf/gHLViw4LL7W1palJ2dfXMXhVsSoQMjlZSUaMyYMdq8ebO2bNniXL6OiopSdna2XnrpJS1cuHCIVwngWmVnZ+utt9761ND5rKs9uH3wGh0Yr7e3Vx9++KEkKTk5WSNHjhziFQG4Xm+++aa6uro0Z86cy+7v6urSgQMHNH369Ju8MtxqCB0AAGAsPhkZAAAYi9ABAADGInQAAICxCB0AAGAsQgfAoM2YMUNlZWVDvYxhbcKECXyWE3AT8Dk6AAbt9ddfvy3fpj9hwgSVlZURecAwQugAGLSkpKQhedyenh7FxMQMyWMDGJ741RWAQfvkr65+/vOfKy0tTXfccYc8Ho/+/M///KrP8cgjj+iRRx7RnXfeqTFjxuhv//ZvIz7NdsKECfrJT36iRYsWybIsFRcXS5K2b9+ue+65R263WxMmTNBTTz0Vce5Lx33nO9/R6NGjNX78eP3qV7/SmTNn9M1vflOjR4/WpEmTdODAgYjjrnTeGTNm6P3339f3v/99uVyuiL+j1NDQoK997WuKjY1VamqqSktL1dXV5ezv6OjQ/PnzFRsbq4kTJ2rbtm1X94UGcP1sABik6dOn248++qi9f/9+Oyoqyn711Vft48eP22+99Zb905/+9KrPMXr0aPvRRx+1Dx8+bL/yyiv2qFGj7Oeee86ZGT9+vJ2QkGBv2LDBPnr0qH306FH7wIED9ogRI+y1a9faR44csV944QU7NjbWfuGFFyKOS0pKsp999ln7vffes7/3ve/Z8fHx9pw5c+xf/vKX9pEjR+wFCxbYGRkZ9sWLF23btj/zvB999JF911132WvXrrXb29vt9vZ227Zt++2337ZHjx5tb9682X7vvffs//qv/7InT55sL1q0yFnP3Llz7aysLLuhocE+cOCAPXXqVDs2NtbevHnz9f2DAPCZCB0Ag3YpdLZv324nJCTYnZ2d13SOT4aGbdv2qlWr7IyMDOf++PHj7QULFkQcV1RUZM+ePTti21//9V/bmZmZEcf95V/+pXO/vb3dlmT/8Ic/dLY1Njbakpxgudrz9o8Tv99vL1myJGLbm2++aY8YMcLu7u62jxw5Ykuym5qanP2HDh2yJRE6wE3Ar64AXLPZs2dr/Pjxuvvuu+X3+7Vt2zb9/ve/v+rjp0yZEvEroNzcXB09etT5I6yS9MADD0Qcc+jQIU2bNi1i27Rp0wYcd++99zr/2+PxSJImTZo0YFtHR8egzttfc3Oztm7dqtGjRzu3/Px8Xbx4UceOHdOhQ4cUHR0d8Tz++I//WHfeeeennhPAjUPoALhm8fHxeuutt/Taa69p3Lhx+tGPfqT77rtP586du2GPERcXF3Hftu2IOLq0rb9Pvivs0vzltl28eHFQ5+3v4sWLWrp0qVpaWpzb7373Ox09elRf/OIXnXP0PzeAm4PQAXBdoqOjNWvWLK1fv15vv/22jh8/rt/85jdXdWxTU9OA+2lpaYqKivrUYzIzM1VfXx+xraGhQX/0R390xeM+y9WcNyYmZsDVnfvvv18HDx7Ul770pQG3mJgYZWRk6OOPP4544fORI0duaAwC+HSEDoBr9utf/1o/+9nP1NLSovfff18vvfSSLl68qPT09Ks6vq2tTStWrNCRI0f02muv6e///u/16KOPXvGY8vJy/fu//7v+7u/+Tu+9955efPFFVVVVqaKi4rqey9Wcd8KECfrtb3+rDz74QB9++KEkadWqVWpsbNSyZcvU0tKio0eP6o033tDy5cslSenp6ZozZ46Ki4u1d+9eNTc367vf/a5iY2Ova70Arg6hA+Ca3XnnnXr99df1jW98QxkZGXr22Wf12muv6Z577rmq47/zne+ou7tbf/qnf6ply5Zp+fLlWrJkyRWPuf/++/XLX/5S1dXVysrK0o9+9COtXbtWixYtuq7ncjXnXbt2rY4fP64vfvGLGjt2rKQ/vBZoz549Onr0qL761a9q8uTJ+uEPf6hx48Y5x73wwgtKTU3V9OnT9a1vfUtLlixRSkrKda0XwNVx2VfzS2gAuMFmzJihP/mTP+HPIAD4XHFFBwAAGIs/AQHghjtx4oQyMzM/df+77757E1cD4HbGr64A3HAff/yxjh8//qn7J0yYoOho/jsLwOeP0AEAAMbiNToAAMBYhA4AADAWoQMAAIxF6AAAAGMROgAAwFiEDgAAMBahAwAAjPX/4vU8uTg6q9YAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "train['is_promoted'].value_counts().plot(kind='bar')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "63aceed8-a1a0-4566-a878-18501b9c19fd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>count</th>\n",
       "      <th>mean</th>\n",
       "      <th>std</th>\n",
       "      <th>min</th>\n",
       "      <th>25%</th>\n",
       "      <th>50%</th>\n",
       "      <th>75%</th>\n",
       "      <th>max</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>employee_id</th>\n",
       "      <td>54808.0</td>\n",
       "      <td>39195.830627</td>\n",
       "      <td>22586.581449</td>\n",
       "      <td>1.0</td>\n",
       "      <td>19669.75</td>\n",
       "      <td>39225.5</td>\n",
       "      <td>58730.5</td>\n",
       "      <td>78298.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>no_of_trainings</th>\n",
       "      <td>54808.0</td>\n",
       "      <td>1.253011</td>\n",
       "      <td>0.609264</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.00</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>10.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>age</th>\n",
       "      <td>54808.0</td>\n",
       "      <td>34.803915</td>\n",
       "      <td>7.660169</td>\n",
       "      <td>20.0</td>\n",
       "      <td>29.00</td>\n",
       "      <td>33.0</td>\n",
       "      <td>39.0</td>\n",
       "      <td>60.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>previous_year_rating</th>\n",
       "      <td>50684.0</td>\n",
       "      <td>3.329256</td>\n",
       "      <td>1.259993</td>\n",
       "      <td>1.0</td>\n",
       "      <td>3.00</td>\n",
       "      <td>3.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>5.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>length_of_service</th>\n",
       "      <td>54808.0</td>\n",
       "      <td>5.865512</td>\n",
       "      <td>4.265094</td>\n",
       "      <td>1.0</td>\n",
       "      <td>3.00</td>\n",
       "      <td>5.0</td>\n",
       "      <td>7.0</td>\n",
       "      <td>37.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>KPIs_met &gt;80%</th>\n",
       "      <td>54808.0</td>\n",
       "      <td>0.351974</td>\n",
       "      <td>0.477590</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.00</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>awards_won?</th>\n",
       "      <td>54808.0</td>\n",
       "      <td>0.023172</td>\n",
       "      <td>0.150450</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.00</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>avg_training_score</th>\n",
       "      <td>54808.0</td>\n",
       "      <td>63.386750</td>\n",
       "      <td>13.371559</td>\n",
       "      <td>39.0</td>\n",
       "      <td>51.00</td>\n",
       "      <td>60.0</td>\n",
       "      <td>76.0</td>\n",
       "      <td>99.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>is_promoted</th>\n",
       "      <td>54808.0</td>\n",
       "      <td>0.085170</td>\n",
       "      <td>0.279137</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.00</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                        count          mean           std   min       25%  \\\n",
       "employee_id           54808.0  39195.830627  22586.581449   1.0  19669.75   \n",
       "no_of_trainings       54808.0      1.253011      0.609264   1.0      1.00   \n",
       "age                   54808.0     34.803915      7.660169  20.0     29.00   \n",
       "previous_year_rating  50684.0      3.329256      1.259993   1.0      3.00   \n",
       "length_of_service     54808.0      5.865512      4.265094   1.0      3.00   \n",
       "KPIs_met >80%         54808.0      0.351974      0.477590   0.0      0.00   \n",
       "awards_won?           54808.0      0.023172      0.150450   0.0      0.00   \n",
       "avg_training_score    54808.0     63.386750     13.371559  39.0     51.00   \n",
       "is_promoted           54808.0      0.085170      0.279137   0.0      0.00   \n",
       "\n",
       "                          50%      75%      max  \n",
       "employee_id           39225.5  58730.5  78298.0  \n",
       "no_of_trainings           1.0      1.0     10.0  \n",
       "age                      33.0     39.0     60.0  \n",
       "previous_year_rating      3.0      4.0      5.0  \n",
       "length_of_service         5.0      7.0     37.0  \n",
       "KPIs_met >80%             0.0      1.0      1.0  \n",
       "awards_won?               0.0      0.0      1.0  \n",
       "avg_training_score       60.0     76.0     99.0  \n",
       "is_promoted               0.0      0.0      1.0  "
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.describe().T"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "c59dbcdd-d769-485a-aed9-f1b56b70d73b",
   "metadata": {},
   "outputs": [],
   "source": [
    "tgt_col = ['is_promoted']\n",
    "ign_cols = ['employee_id']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "29b6bd74-bb34-460b-925e-380ededf0eed",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>count</th>\n",
       "      <th>mean</th>\n",
       "      <th>std</th>\n",
       "      <th>min</th>\n",
       "      <th>25%</th>\n",
       "      <th>50%</th>\n",
       "      <th>75%</th>\n",
       "      <th>max</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>no_of_trainings</th>\n",
       "      <td>54808.0</td>\n",
       "      <td>1.253011</td>\n",
       "      <td>0.609264</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>10.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>age</th>\n",
       "      <td>54808.0</td>\n",
       "      <td>34.803915</td>\n",
       "      <td>7.660169</td>\n",
       "      <td>20.0</td>\n",
       "      <td>29.0</td>\n",
       "      <td>33.0</td>\n",
       "      <td>39.0</td>\n",
       "      <td>60.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>previous_year_rating</th>\n",
       "      <td>50684.0</td>\n",
       "      <td>3.329256</td>\n",
       "      <td>1.259993</td>\n",
       "      <td>1.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>5.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>length_of_service</th>\n",
       "      <td>54808.0</td>\n",
       "      <td>5.865512</td>\n",
       "      <td>4.265094</td>\n",
       "      <td>1.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>7.0</td>\n",
       "      <td>37.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>KPIs_met &gt;80%</th>\n",
       "      <td>54808.0</td>\n",
       "      <td>0.351974</td>\n",
       "      <td>0.477590</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>awards_won?</th>\n",
       "      <td>54808.0</td>\n",
       "      <td>0.023172</td>\n",
       "      <td>0.150450</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>avg_training_score</th>\n",
       "      <td>54808.0</td>\n",
       "      <td>63.386750</td>\n",
       "      <td>13.371559</td>\n",
       "      <td>39.0</td>\n",
       "      <td>51.0</td>\n",
       "      <td>60.0</td>\n",
       "      <td>76.0</td>\n",
       "      <td>99.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>is_promoted</th>\n",
       "      <td>54808.0</td>\n",
       "      <td>0.085170</td>\n",
       "      <td>0.279137</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                        count       mean        std   min   25%   50%   75%  \\\n",
       "no_of_trainings       54808.0   1.253011   0.609264   1.0   1.0   1.0   1.0   \n",
       "age                   54808.0  34.803915   7.660169  20.0  29.0  33.0  39.0   \n",
       "previous_year_rating  50684.0   3.329256   1.259993   1.0   3.0   3.0   4.0   \n",
       "length_of_service     54808.0   5.865512   4.265094   1.0   3.0   5.0   7.0   \n",
       "KPIs_met >80%         54808.0   0.351974   0.477590   0.0   0.0   0.0   1.0   \n",
       "awards_won?           54808.0   0.023172   0.150450   0.0   0.0   0.0   0.0   \n",
       "avg_training_score    54808.0  63.386750  13.371559  39.0  51.0  60.0  76.0   \n",
       "is_promoted           54808.0   0.085170   0.279137   0.0   0.0   0.0   0.0   \n",
       "\n",
       "                       max  \n",
       "no_of_trainings       10.0  \n",
       "age                   60.0  \n",
       "previous_year_rating   5.0  \n",
       "length_of_service     37.0  \n",
       "KPIs_met >80%          1.0  \n",
       "awards_won?            1.0  \n",
       "avg_training_score    99.0  \n",
       "is_promoted            1.0  "
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.drop(columns=ign_cols).describe().T"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "3e6c59c2-ff05-4dde-9fe6-c058399ea121",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>count</th>\n",
       "      <th>unique</th>\n",
       "      <th>top</th>\n",
       "      <th>freq</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>department</th>\n",
       "      <td>54808</td>\n",
       "      <td>9</td>\n",
       "      <td>Sales &amp; Marketing</td>\n",
       "      <td>16840</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>region</th>\n",
       "      <td>54808</td>\n",
       "      <td>34</td>\n",
       "      <td>region_2</td>\n",
       "      <td>12343</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>education</th>\n",
       "      <td>52399</td>\n",
       "      <td>3</td>\n",
       "      <td>Bachelor's</td>\n",
       "      <td>36669</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>gender</th>\n",
       "      <td>54808</td>\n",
       "      <td>2</td>\n",
       "      <td>m</td>\n",
       "      <td>38496</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>recruitment_channel</th>\n",
       "      <td>54808</td>\n",
       "      <td>3</td>\n",
       "      <td>other</td>\n",
       "      <td>30446</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                     count unique                top   freq\n",
       "department           54808      9  Sales & Marketing  16840\n",
       "region               54808     34           region_2  12343\n",
       "education            52399      3         Bachelor's  36669\n",
       "gender               54808      2                  m  38496\n",
       "recruitment_channel  54808      3              other  30446"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.describe(include='object').T"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "1df84478-d8ec-4eaa-a167-c81ceff61107",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "employee_id             54808\n",
       "department                  9\n",
       "region                     34\n",
       "education                   3\n",
       "gender                      2\n",
       "recruitment_channel         3\n",
       "no_of_trainings            10\n",
       "age                        41\n",
       "previous_year_rating        5\n",
       "length_of_service          35\n",
       "KPIs_met >80%               2\n",
       "awards_won?                 2\n",
       "avg_training_score         61\n",
       "is_promoted                 2\n",
       "dtype: int64"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.nunique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "881551d9-2ab8-489a-afe4-05ef77e331e0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "department 9 => ['Sales & Marketing' 'Operations' 'Technology' 'Analytics' 'R&D'\n",
      " 'Procurement' 'Finance' 'HR' 'Legal']\n",
      "region 34 => ['region_7' 'region_22' 'region_19' 'region_23' 'region_26' 'region_2'\n",
      " 'region_20' 'region_34' 'region_1' 'region_4' 'region_29' 'region_31'\n",
      " 'region_15' 'region_14' 'region_11' 'region_5' 'region_28' 'region_17'\n",
      " 'region_13' 'region_16' 'region_25' 'region_10' 'region_27' 'region_30'\n",
      " 'region_12' 'region_21' 'region_8' 'region_32' 'region_6' 'region_33'\n",
      " 'region_24' 'region_3' 'region_9' 'region_18']\n",
      "education 3 => [\"Master's & above\" \"Bachelor's\" nan 'Below Secondary']\n",
      "gender 2 => ['f' 'm']\n",
      "recruitment_channel 3 => ['sourcing' 'other' 'referred']\n",
      "no_of_trainings 10 => [ 1  2  3  4  7  5  6  8 10  9]\n",
      "age 41 => [35 30 34 39 45 31 33 28 32 49 37 38 41 27 29 26 24 57 40 42 23 59 44 50\n",
      " 56 20 25 47 36 46 60 43 22 54 58 48 53 55 51 52 21]\n",
      "previous_year_rating 5 => [ 5.  3.  1.  4. nan  2.]\n",
      "length_of_service 35 => [ 8  4  7 10  2  5  6  1  3 16  9 11 26 12 17 14 13 19 15 23 18 20 22 25\n",
      " 28 24 31 21 29 30 34 27 33 32 37]\n",
      "KPIs_met >80% 2 => [1 0]\n",
      "awards_won? 2 => [0 1]\n",
      "avg_training_score 61 => [49 60 50 73 85 59 63 83 54 77 80 84 51 46 75 57 70 68 79 44 72 61 48 58\n",
      " 87 47 52 88 71 65 62 53 78 91 82 69 55 74 86 90 92 67 89 56 76 81 45 64\n",
      " 39 94 93 66 95 42 96 40 99 43 97 41 98]\n",
      "is_promoted 2 => [0 1]\n"
     ]
    }
   ],
   "source": [
    "# Droping employee ID\n",
    "for col in train.drop(columns=ign_cols).columns:\n",
    "    print(col,train[col].nunique(),  '=>', train[col].unique())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "b7e68633-2fb4-406a-bf7f-5361c3ed0d50",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Replacing nan with 0 in previous year rating - i.e freshers\n",
    "train['previous_year_rating'] = train['previous_year_rating'].fillna(0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "08b5ba2a-1222-41b9-8fee-ce08c81e266b",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "employee_id 54808 => [65438 65141  7513 ... 13918 13614 51526]\n",
      "department 9 => ['Sales & Marketing' 'Operations' 'Technology' 'Analytics' 'R&D'\n",
      " 'Procurement' 'Finance' 'HR' 'Legal']\n",
      "region 34 => ['region_7' 'region_22' 'region_19' 'region_23' 'region_26' 'region_2'\n",
      " 'region_20' 'region_34' 'region_1' 'region_4' 'region_29' 'region_31'\n",
      " 'region_15' 'region_14' 'region_11' 'region_5' 'region_28' 'region_17'\n",
      " 'region_13' 'region_16' 'region_25' 'region_10' 'region_27' 'region_30'\n",
      " 'region_12' 'region_21' 'region_8' 'region_32' 'region_6' 'region_33'\n",
      " 'region_24' 'region_3' 'region_9' 'region_18']\n",
      "education 4 => [\"Master's & above\" \"Bachelor's\" 'unschooled' 'Below Secondary']\n",
      "gender 2 => ['f' 'm']\n",
      "recruitment_channel 3 => ['sourcing' 'other' 'referred']\n",
      "no_of_trainings 10 => [ 1  2  3  4  7  5  6  8 10  9]\n",
      "age 41 => [35 30 34 39 45 31 33 28 32 49 37 38 41 27 29 26 24 57 40 42 23 59 44 50\n",
      " 56 20 25 47 36 46 60 43 22 54 58 48 53 55 51 52 21]\n",
      "previous_year_rating 6 => [5. 3. 1. 4. 0. 2.]\n",
      "length_of_service 35 => [ 8  4  7 10  2  5  6  1  3 16  9 11 26 12 17 14 13 19 15 23 18 20 22 25\n",
      " 28 24 31 21 29 30 34 27 33 32 37]\n",
      "KPIs_met >80% 2 => [1 0]\n",
      "awards_won? 2 => [0 1]\n",
      "avg_training_score 61 => [49 60 50 73 85 59 63 83 54 77 80 84 51 46 75 57 70 68 79 44 72 61 48 58\n",
      " 87 47 52 88 71 65 62 53 78 91 82 69 55 74 86 90 92 67 89 56 76 81 45 64\n",
      " 39 94 93 66 95 42 96 40 99 43 97 41 98]\n",
      "is_promoted 2 => [0 1]\n"
     ]
    }
   ],
   "source": [
    "for col in train.columns:\n",
    "    print(col,train[col].nunique(),  '=>', train[col].unique())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "d5c39d2a-eb0b-430d-873b-2adb7a185399",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 4.39% education contains nan. We can replace this with unschooled\n",
    "train['education'] = train['education'].fillna('unschooled')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "e61a0ff7-b928-481e-ac8c-ee2d95c7eeb5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='no_of_trainings', ylabel='Density'>"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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gdZONrfK/Pe5pV6J4ZLt27cbkox333232eO//OUvjeOPP77Vr7n77rsNNfwPKt5444033njj7Qh/KysrS5kXXOuoHHXUUQoEAod1TyoqKg7rsphuv/12zZ07N/5xLBbT3r171bdvX9bFbaiqqlJxcbHKyspcXTrrrnj93cXr7y5ef3e5/fobhqHq6moNGDAg5XNdCyqZmZk65ZRTtHz58mYzKsuXL9e0adNa/ZpQKKRQKNTssV69enVkmd1CXl4e/6JwEa+/u3j93cXr7y43X//8/HxLz3MtqEjS3LlzdcUVV2jcuHEaP368Fi1apG3btulHP/qRm2UBAACPcDWoXHrppdqzZ4/uuecelZeXa+TIkfrrX/+qY445xs2yAACAR7gaVCTp+uuv1/XXX+92Gd1SKBTS3XfffdhyGjoHr7+7eP3dxevvriPp9Xf1wDcAAIC2uH7XDwAAQDIEFQAA4FkEFQAA4FkElW5o3rx5OvXUU5Wbm6uCggJNnz5dmzdvdrusbmvevHny+XyaM2eO26V0G9u3b9d//dd/qW/fvsrJydFJJ52kdes4+L+jRSIR/fznP9eQIUOUnZ2toUOH6p577lEsFnO7tC5pzZo1uvDCCzVgwAD5fD699NJLzT5vGIZ+8YtfaMCAAcrOztbZZ5+tTz75xJ1i20BQ6YZWr16tWbNm6b333tPy5csViUR03nnnqba21u3Sup2SkhItWrRIo0ePdruUbmPfvn2aMGGCMjIy9Nprr+nTTz/V/PnzOTyyE9x///363//9Xz366KPatGmTHnjgAT344IP6n//5H7dL65Jqa2s1ZswYPfroo61+/oEHHtCCBQv06KOPqqSkREVFRTr33HNVXd32nXCdjV0/0O7du1VQUKDVq1dr0qRJbpfTbdTU1Ojkk0/W7373O/3yl7/USSedpIceesjtsrq82267TX//+9/19ttvu11Kt/Pd735XhYWFeuKJJ+KPff/731dOTo6efPJJFyvr+nw+n5YtW6bp06dLauimDBgwQHPmzNGtt94qqeHi38LCQt1///267rrrXKy2OToqiN9C3adPH5cr6V5mzZqlCy64QFOmTHG7lG7l5Zdf1rhx43TxxReroKBAY8eO1eOPP+52Wd3CxIkTtWLFCn322WeSpI8//ljvvPOOzj//fJcr6362bt2qnTt36rzzzos/FgqFdNZZZ+ndd991sbLDuX7gG9xlGIbmzp2riRMnauTIkW6X0208++yz+vDDD1VSUuJ2Kd3Ol19+qYULF2ru3Lm644479MEHH+imm25SKBTSlVde6XZ5Xdqtt96qyspKDR8+XIFAQNFoVPfee6/+8z//0+3Suh3zQuCWlwAXFhaqtLTUjZKSIqh0czfccIM2bNigd955x+1Suo2ysjLNnj1bb7zxhrKystwup9uJxWIaN26c7rvvPknS2LFj9cknn2jhwoUElQ723HPP6amnntIzzzyjESNGaP369ZozZ44GDBigmTNnul1et+Tz+Zp9bBjGYY+5jaDSjd144416+eWXtWbNGg0cONDtcrqNdevWqaKiQqecckr8sWg0qjVr1ujRRx9VOBxWIBBwscKurX///jrxxBObPXbCCSfohRdecKmi7uOnP/2pbrvtNv3Hf/yHJGnUqFEqLS3VvHnzCCqdrKioSFJDZ6V///7xxysqKg7rsriNGZVuyDAM3XDDDXrxxRf11ltvaciQIW6X1K1MnjxZGzdu1Pr16+Nv48aN0+WXX67169cTUjrYhAkTDtuO/9lnn3EZaic4cOCA/P7mf3YCgQDbk10wZMgQFRUVafny5fHH6urqtHr1ap1xxhkuVnY4Oird0KxZs/TMM8/oL3/5i3Jzc+Nrlfn5+crOzna5uq4vNzf3sHmgHj16qG/fvswJdYKf/OQnOuOMM3Tffffpkksu0QcffKBFixZp0aJFbpfW5V144YW69957NWjQII0YMUIfffSRFixYoGuuucbt0rqkmpoaffHFF/GPt27dqvXr16tPnz4aNGiQ5syZo/vuu0/HHXecjjvuON13333KycnRZZdd5mLVrTDQ7Uhq9W3x4sVul9ZtnXXWWcbs2bPdLqPbeOWVV4yRI0caoVDIGD58uLFo0SK3S+oWqqqqjNmzZxuDBg0ysrKyjKFDhxp33nmnEQ6H3S6tS1q5cmWr/66fOXOmYRiGEYvFjLvvvtsoKioyQqGQMWnSJGPjxo3uFt0KzlEBAACexYwKAADwLIIKAADwLIIKAADwLIIKAADwLIIKAADwLIIKAADwLIIKAADwLIIKAADwLIIKAMfs3LlT5557rnr06KFevXp16s/+6quv5PP5tH79estf84tf/EInnXRSh9UEoP04mRaAY2699Va9+uqrWrZsmfLz81VQUNDm830+n5YtW6bp06e3+2dHo1Ht3r1bRx11lIJBa9eY1dTUKBwOq2/fvu3++QA6BpcSAnDMli1bdMopp+i4445z7HvW19crIyMj5fMCgUD86nqrevbsqZ49e9otDUAnYOkH6ELOPvts3XTTTfrZz36mPn36qKioSL/4xS/in9+2bZumTZumnj17Ki8vT5dccol27dpl+fsvXLhQxx57rDIzM3X88cfrySefjH9u8ODBeuGFF7R06VL5fD5dddVVbX6vwYMHS5JmzJghn88X/9hcjvm///s/DR06VKFQSIZh6PXXX9fEiRPVq1cv9e3bV9/97ne1ZcuW+PdrufSzatUq+Xw+rVixQuPGjVNOTo7OOOMMbd68Of41LZd+rrrqKk2fPl2//vWv1b9/f/Xt21ezZs1SfX19/Dnl5eW64IILlJ2drSFDhuiZZ57R4MGD9dBDDzX7voMGDVIoFNKAAQN00003WX6NATRHUAG6mD/84Q/q0aOH3n//fT3wwAO65557tHz5chmGoenTp2vv3r1avXq1li9fri1btujSSy+19H2XLVum2bNn6+abb9Y///lPXXfddbr66qu1cuVKSVJJSYm+853v6JJLLlF5ebkefvjhNr9fSUmJJGnx4sUqLy+PfyxJX3zxhf70pz/phRdeiAeP2tpazZ07VyUlJVqxYoX8fr9mzJihWCzW5s+58847NX/+fK1du1bBYFDXXHNNm89fuXKltmzZopUrV+oPf/iDlixZoiVLlsQ/f+WVV2rHjh1atWqVXnjhBS1atEgVFRXxzz///PP6zW9+o8cee0yff/65XnrpJY0aNarNnwmgDW5e3QzAWWeddZYxceLEZo+deuqpxq233mq88cYbRiAQMLZt2xb/3CeffGJIMj744IOU3/uMM84wrr322maPXXzxxcb5558f/3jatGnxK+StkGQsW7as2WN33323kZGRYVRUVLT5tRUVFYak+LX0W7duNSQZH330kWEYTVfcv/nmm/GvefXVVw1JxsGDB+M/a8yYMfHPz5w50zjmmGOMSCTS7He89NJLDcMwjE2bNhmSjJKSkvjnP//8c0OS8Zvf/MYwDMOYP3++8a1vfcuoq6uz/DoASI6OCtDFjB49utnH/fv3V0VFhTZt2qTi4mIVFxfHP3fiiSeqV69e2rRpU8rvu2nTJk2YMKHZYxMmTLD0tek65phj1K9fv2aPbdmyRZdddpmGDh2qvLw8DRkyRFLDclZbEl+P/v37S1KzDkhLI0aMUCAQaPY15vM3b96sYDCok08+Of75YcOGqXfv3vGPL774Yh08eFBDhw7Vtddeq2XLlikSiaT6lQEkQVABupiWg6c+n0+xWEyGYcjn8x32/GSPt6bl89L52nT06NHjsMcuvPBC7dmzR48//rjef/99vf/++5Kkurq6Nr9X4uth1trWclGy109q+H1bk/h4cXGxNm/erN/+9rfKzs7W9ddfr0mTJjWbcwFgHUEF6CZOPPFEbdu2TWVlZfHHPv30U1VWVuqEE05I+fUnnHCC3nnnnWaPvfvuu5a+NpmMjAxFo9GUz9uzZ482bdqkn//855o8ebJOOOEE7du3z/bPtWv48OGKRCL66KOP4o998cUX2r9/f7PnZWdn63vf+54eeeQRrVq1Sv/4xz+0cePGTq4W6BrYngx0E1OmTNHo0aN1+eWX66GHHlIkEtH111+vs846S+PGjUv59T/96U91ySWX6OSTT9bkyZP1yiuv6MUXX9Sbb75pu6bBgwdrxYoVmjBhgkKhULMllES9e/dW3759tWjRIvXv31/btm3TbbfdZvvn2jV8+HBNmTJFP/zhD7Vw4UJlZGTo5ptvVnZ2drxbs2TJEkWjUZ1++unKycnRk08+qezsbB1zzDGdXi/QFdBRAboJn8+nl156Sb1799akSZM0ZcoUDR06VM8995ylr58+fboefvhhPfjggxoxYoQee+wxLV68WGeffbbtmubPn6/ly5eruLhYY8eOTfo8v9+vZ599VuvWrdPIkSP1k5/8RA8++KDtn9seS5cuVWFhoSZNmqQZM2bo2muvVW5urrKysiRJvXr10uOPP64JEyZo9OjRWrFihV555RUOlQNs4mRaAGiHr7/+WsXFxXrzzTc1efJkt8sBuhyCCgCk4a233lJNTY1GjRql8vJy/exnP9P27dv12WefWTpBF0B6WPoBIKlhW655pHzLt6effjrt7/f0008n/X4jRozogN+gc9TX1+uOO+7QiBEjNGPGDPXr10+rVq0ipAAdhI4KAElSaWlp0i20hYWFys3NTev7VVdXJz2ePyMjg+FSAJYQVAAAgGex9AMAADyLoAIAADyLoAIAADyLoAIAADyLoAIAADyLoAIAADyLoAIAADyLoAIAADzr/wOfhABVwcXH+gAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.distplot(train.no_of_trainings)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "0ea4ab5d-d4f5-4ba5-8633-7810fd165f4d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='avg_training_score', ylabel='Density'>"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAkAAAAGxCAYAAACKvAkXAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy81sbWrAAAACXBIWXMAAA9hAAAPYQGoP6dpAABo6ElEQVR4nO3dd3hUVfoH8O+0zKT33kOAREIzoQRErLhgAWzYdRHcLO4qZF0buq5Y2F2R5ceugLogixVXlNU1KgEBC5ESEmoIENJI7z1T7++PyQyEBEiZzJ3y/TzPPMKdc++89xqSN+e85xyJIAgCiIiIiJyIVOwAiIiIiKyNCRARERE5HSZARERE5HSYABEREZHTYQJERERETocJEBERETkdJkBERETkdJgAERERkdORix2ALTIYDCgvL4enpyckEonY4RAREVEfCIKAlpYWhIWFQSq9dB8PE6BelJeXIzIyUuwwiIiIaABKS0sRERFxyTZMgHrh6ekJwPgAvby8RI6GiIiI+qK5uRmRkZHmn+OXwgSoF6ZhLy8vLyZAREREdqYv5SssgiYiIiKnwwSIiIiInA4TICIiInI6oidAa9asQWxsLFQqFZKTk/Hjjz9esv3u3buRnJwMlUqFuLg4rFu3rkebxsZGPP744wgNDYVKpUJiYiIyMjKG6haIiIjIzoiaAG3evBmLFy/G0qVLkZOTg2nTpmHmzJkoKSnptX1hYSFmzZqFadOmIScnB88//zyeeOIJbNmyxdxGo9HgxhtvRFFRET777DPk5+fj3XffRXh4uLVui4iIiGycRBAEQawPnzRpEq688kqsXbvWfCwxMRFz5szB8uXLe7R/5pln8OWXXyIvL898LC0tDYcOHUJWVhYAYN26dXjjjTdw4sQJKBSKAcXV3NwMb29vNDU1cRYYERGRnejPz2/ReoA0Gg2ys7MxY8aMbsdnzJiBPXv29HpOVlZWj/Y33XQTDhw4AK1WCwD48ssvkZqaiscffxzBwcFISkrC66+/Dr1ef9FY1Go1mpubu72IiIjIcYmWANXW1kKv1yM4OLjb8eDgYFRWVvZ6TmVlZa/tdTodamtrAQBnzpzBZ599Br1ej4yMDLzwwgt488038dprr100luXLl8Pb29v84irQREREjk30IugLFysSBOGSCxj11v784waDAUFBQXjnnXeQnJyMe+65B0uXLu02zHah5557Dk1NTeZXaWnpQG+HiIiI7IBoK0EHBARAJpP16O2prq7u0ctjEhIS0mt7uVwOf39/AEBoaCgUCgVkMpm5TWJiIiorK6HRaODi4tLjukqlEkqlcrC3RERERHZCtB4gFxcXJCcnIzMzs9vxzMxMTJkypddzUlNTe7Tftm0bUlJSzAXPU6dOxenTp2EwGMxtTp48idDQ0F6THyIiInI+og6Bpaen41//+hc2bNiAvLw8LFmyBCUlJUhLSwNgHJp66KGHzO3T0tJQXFyM9PR05OXlYcOGDVi/fj2eeuopc5vf/va3qKurw5NPPomTJ0/i66+/xuuvv47HH3/c6vdHREREtknUzVDnzZuHuro6LFu2DBUVFUhKSkJGRgaio6MBABUVFd3WBIqNjUVGRgaWLFmCt956C2FhYVi9ejXuuOMOc5vIyEhs27YNS5YswZgxYxAeHo4nn3wSzzzzjNXvj4iIiGyTqOsA2SquA0RERGR/7GIdICIiIiKxiDoERmTLPtrb+5Ys57tvUpQVIiEiIktjAkR2h4kJERENFofAiIiIyOkwASIiIiKnwwSIiIiInA4TICIiInI6TICIiIjI6TABIiIiIqfDBIiIiIicDhMgIiIicjpMgIiIiMjpMAEiIiIip8MEiIiIiJwOEyAiIiJyOtwMlYgAXH6TWW4wS0SOhD1ARERE5HSYABEREZHTYQJERERETocJEBERETkdJkBERETkdJgAERERkdNhAkREREROhwkQEREROR0mQEREROR0mAARERGR02ECRERERE6HCRARERE5HSZARERE5HSYABEREZHTYQJERERETocJEBERETkdJkBERETkdJgAERERkdNhAkREREROhwkQEREROR0mQEREROR0mAARERGR02ECRERERE6HCRARERE5HSZARERE5HSYABEREZHTYQJERERETocJEBERETkdJkBERETkdJgAERERkdNhAkREREROhwkQEREROR0mQEREROR0RE+A1qxZg9jYWKhUKiQnJ+PHH3+8ZPvdu3cjOTkZKpUKcXFxWLduXbf3N27cCIlE0uPV2dk5lLdBREREdkTUBGjz5s1YvHgxli5dipycHEybNg0zZ85ESUlJr+0LCwsxa9YsTJs2DTk5OXj++efxxBNPYMuWLd3aeXl5oaKiottLpVJZ45aIiIjIDsjF/PCVK1fi0UcfxYIFCwAAq1atwnfffYe1a9di+fLlPdqvW7cOUVFRWLVqFQAgMTERBw4cwIoVK3DHHXeY20kkEoSEhFjlHoiIiMj+iNYDpNFokJ2djRkzZnQ7PmPGDOzZs6fXc7Kysnq0v+mmm3DgwAFotVrzsdbWVkRHRyMiIgK33HILcnJyLhmLWq1Gc3NztxcRERE5LtESoNraWuj1egQHB3c7HhwcjMrKyl7Pqays7LW9TqdDbW0tACAhIQEbN27El19+iY8//hgqlQpTp07FqVOnLhrL8uXL4e3tbX5FRkYO8u6IiIjIloleBC2RSLr9XRCEHscu1/7845MnT8YDDzyAsWPHYtq0afj0008xYsQI/OMf/7joNZ977jk0NTWZX6WlpQO9HSIiIrIDotUABQQEQCaT9ejtqa6u7tHLYxISEtJre7lcDn9//17PkUqlmDBhwiV7gJRKJZRKZT/vgIiIiOyVaD1ALi4uSE5ORmZmZrfjmZmZmDJlSq/npKam9mi/bds2pKSkQKFQ9HqOIAjIzc1FaGioZQInIiIiuyfqEFh6ejr+9a9/YcOGDcjLy8OSJUtQUlKCtLQ0AMahqYceesjcPi0tDcXFxUhPT0deXh42bNiA9evX46mnnjK3efnll/Hdd9/hzJkzyM3NxaOPPorc3FzzNYmIiIhEnQY/b9481NXVYdmyZaioqEBSUhIyMjIQHR0NAKioqOi2JlBsbCwyMjKwZMkSvPXWWwgLC8Pq1au7TYFvbGzEY489hsrKSnh7e2P8+PH44YcfMHHiRKvfHxEREdkmiWCqIiaz5uZmeHt7o6mpCV5eXmKHQxf4aG/vC2We775JUXbzObbicvfrSPdKRI6pPz+/RZ8FRkRERGRtTICIiIjI6TABIiIiIqfDBIiIiIicDhMgIiIicjpMgIiIiMjpMAEiIiIip8MEiIiIiJwOEyAiIiJyOkyAiIiIyOkwASIiIiKnwwSIiIiInA4TICIiInI6TICIiIjI6TABIiIiIqfDBIiIiIicDhMgIiIicjpMgIiIiMjpMAEiIiIip8MEiIiIiJwOEyAiIiJyOkyAiIiIyOkwASIiIiKnwwSIiIiInA4TICIiInI6TICIiIjI6TABIiIiIqfDBIiIiIicjlzsAIis7c1t+fgipwzhPq4YGeKJRdfEI8RbJXZYRERkRUyAyKn850Ap/vH9aQDA2YYO7C2sR25pIz7/7RTIZewQJSJyFvyOT07jaFkTlm49CgB49KpYrLx7LLxUchw+24S3fzgjcnRERGRN7AEip6DRGfDbD7Oh0RlwfUIQls5KhFQqgSAAf/jPIfzf9lO48YpgjAj2FDtUIiKyAvYAkVP4/kQ1Sus7EOChxMp54yCVSgAAt18ZjusSgqDRG/DSf4+JHCUREVkLEyByCp9lnwUA3JEcDm9Xhfm4RCLBK3OSIJUAWWfqUFLXLlaIRERkRUyAyOHVtKixM78aAHBXckSP98N9XDE1PgAAsDW3zKqxERGROJgAkcP7b24Z9AYB4yJ9EB/Ue43P3PHhAIAvcsogCII1wyMiIhEwASKHJgiCefjrzl56f0xuGhUCV4UMhbVtyC1ttFJ0REQkFiZA5NCOlTfjRGULXORS3Dom7KLt3JVy3DQqGICxF4iIiBwbEyByaLtP1gAArhkRCG83xSXbzr3S2EP01aFyaPWGIY+NiIjEwwSIHNovZ+oAAFOG+V+27dRh/vBzd0FDuxaHOAxGROTQmACRw9LqDThQ1AAAmNyHBEgukyI1zthuT0HdkMZGRETiYgJEDuvw2SZ0aPXwc3fBiIvM/rpQ6jBTAlQ7lKEREZHImACRwzINf02K9TOv/Hw5pqGyg8WNrAMiInJgTIDIYZkSoMlxlx/+MokNcEeIlwoavQHFXBWaiMhhMQEih6TRnVf/048ESCKRmHuBCmpahyQ2IiISHxMgckhHyhrN9T/Dgzz6da6pDugMEyAiIofFBIgc0i9n6gH0r/7HZErXvmBnGzrQqdVbPDYiIhIfEyBySDkljQCAlBi/fp8b7uOKGH83CACKatssGxgREdkEJkDkkI6XNwEAksK8BnT+hK7EqaSBhdBERI5I9ARozZo1iI2NhUqlQnJyMn788cdLtt+9ezeSk5OhUqkQFxeHdevWXbTtJ598AolEgjlz5lg4arJlbWodyps6AQBXDDABGhPpAwAoa+iwVFhERGRDRE2ANm/ejMWLF2Pp0qXIycnBtGnTMHPmTJSUlPTavrCwELNmzcK0adOQk5OD559/Hk888QS2bNnSo21xcTGeeuopTJs2bahvg2xMeZMxaYnxd4On6tL7f13MuAgfAMY6IEEQLBUaERHZCFEToJUrV+LRRx/FggULkJiYiFWrViEyMhJr167ttf26desQFRWFVatWITExEQsWLMD8+fOxYsWKbu30ej3uv/9+vPzyy4iLi7PGrZANqWg09v6MCvMe8DVGhnhCJpWgQ6tHfZvGUqEREZGNEC0B0mg0yM7OxowZM7odnzFjBvbs2dPrOVlZWT3a33TTTThw4AC0Wq352LJlyxAYGIhHH320T7Go1Wo0Nzd3e5H9Kms09gCNCh/Y8BcAuMilCPVWATD2AhERkWMRLQGqra2FXq9HcHBwt+PBwcGorKzs9ZzKyspe2+t0OtTWGvdu+vnnn7F+/Xq8++67fY5l+fLl8Pb2Nr8iIyP7eTdkSyq6hsAG0wMEABG+bgCAsyyEJiJyOKIXQUsk3ddoEQShx7HLtTcdb2lpwQMPPIB3330XAQEBfY7hueeeQ1NTk/lVWlrajzsgW6LW6lHXahyyGjXAAmiTCF9XAOwBIiJyRHKxPjggIAAymaxHb091dXWPXh6TkJCQXtvL5XL4+/vj2LFjKCoqwq233mp+32Awbmgpl8uRn5+PYcOG9biuUqmEUqkc7C2RDaho6oQAIMRLhQCPwf0/NSVA5U0d0BsEyPq5oCIREdku0XqAXFxckJycjMzMzG7HMzMzMWXKlF7PSU1N7dF+27ZtSElJgUKhQEJCAo4cOYLc3Fzz67bbbsO1116L3NxcDm05gXLz8Nfgen8AIMBDCaVcCq1eQHVL56CvR0REtkO0HiAASE9Px4MPPoiUlBSkpqbinXfeQUlJCdLS0gAYh6bKysqwadMmAEBaWhr++c9/Ij09HQsXLkRWVhbWr1+Pjz/+GACgUqmQlJTU7TN8fHwAoMdxckzmGWDhg6v/AQCpRIJwX1ecqWnD2foOhHq7DvqaRERkG0RNgObNm4e6ujosW7YMFRUVSEpKQkZGBqKjowEAFRUV3dYEio2NRUZGBpYsWYK33noLYWFhWL16Ne644w6xboFsjCV7gAAgwsfNmAA1tmMC+r+tBpGj+Ghv7+uzmdw3KcpKkRBZhqgJEAAsWrQIixYt6vW9jRs39jg2ffp0HDx4sM/X7+0a5JgMgoCaFjUAICHE0yLXDPMxToWvaOIQGBGRIxF9FhiRpTS0aaAzCJBLJeYp7IMV0rUWUFVzJwxcEZqIyGEwASKHYer9CfBQWmzGVoCHEgqZBFq9gPpWrghNROQomACRw6juSoCCvCy3pIFUIkGwV9cwWDOHwYiIHAUTIHIYpgQo0NOyazqFmBKgJi6ISETkKJgAkcOo6VqrJ8hTZdHrmuqAKlkITUTkMESfBUZkCYIg9LsH6HLTek2YABEROR72AJFDaOnUQa0zQAIgwN3FotcO9TIugNjYoUWHRm/RaxMRkTiYAJFDMPX++Hu4QC6z7Je1q4sMPq4KAEAlC6GJiBwCh8DIIZjqfwK76n/6OrzVVyHeKjR2aFHR1IHYAHeLXpuIiKyPPUDkEMxT4C08A8yEdUBERI6FCRA5hKGaAm9i2giVQ2BERI6BCRA5hJoh7gEK7lpckVtiEBE5BiZAZPc6NHq0qnUAgECPoUmA/N2VkEmMW2I0tmuH5DOIiMh6mACR3attNfb+eKnkUCpkQ/IZMqkEAZ7G6fXVLRwGIyKyd0yAyO7VtZmmwA9N74+JaYXp6mb1kH4OERENPSZAZPdqu3Zp97fwAogXMm2yyh4gIiL7xwSI7F59W1cCNMQ9QMGmHqAW9gAREdk7JkBk90w1QEPeA9Q1w6y6WQ2BM8GIiOwaEyCye3WmITCPoU2A/D2MM8E0egMaOzgTjIjInjEBIrvWrtGhQ2vcoNTffWiHwGRSiTnJYiE0EZF9YwJEds3U++OlksNFPvRfzkFepjogFkITEdkzJkBk1+qsVABtcn4dEBER2S8mQGTX6qxUAG0SzB4gIiKHIBc7AKLBEK0HqMV6M8E+2lty2Tb3TYrq93U7NHpoDQZ4qRQDCYuIyK4NKAEqLCxEbGyspWMh6jdr9wD5e7hAKgHUOgOa7HQmmE5vwHs/F+HNzHxo9QKuGRGIeyf2P4EiIrJnAxoCi4+Px7XXXosPPvgAnZ0cCiDxnOsBsk4CJJdKzb1N9rggYl2rGnes3YPXMvLQqTVAbxCw40Q1Fmw6gANF9WKHR0RkNQNKgA4dOoTx48fjD3/4A0JCQvCb3/wG+/bts3RsRJfUodGjXWOdKfDnO38YzN78+avjOHS2CV4qOf52xxhsT78a90yIBAB8dbictU1E5DQGlAAlJSVh5cqVKCsrw3vvvYfKykpcddVVGDVqFFauXImamhpLx0nUg2kTVE8rTYE3Obcpqn0lC7tP1uCrQ+WQSoAPF0zG3RMiER/kidfnjsZV8QHQ6gVs3l8Knd4gdqhERENuUD815HI55s6di08//RR//etfUVBQgKeeegoRERF46KGHUFFRYak4iXqw1iaoFwr2sr8eoE6tHi9uPQoAeGRKLEZHeJvfk0olWHn3WLi5yFDR1IkfTtWKFSYRkdUMKgE6cOAAFi1ahNDQUKxcuRJPPfUUCgoK8P3336OsrAyzZ8+2VJxEPZg3QbXi8BdwXg9QS6fd7An2zg9nUFLfjlBvFdJnjOjxfpCXCjePDgUA7D1TB52BvUBE5NgGlACtXLkSo0ePxpQpU1BeXo5NmzahuLgYr776KmJjYzF16lS8/fbbOHjwoKXjJTJr6EqAfK3cAxTQNROsU2tAlR0siKjVG/D+L8UAgGd+lQAPZe+TP8dE+MBTJUeLWodjZc3WDJGIyOoGlACtXbsW9913H0pKSrB161bccsstkEq7XyoqKgrr16+3SJBEvalvNyZAfu7WXcdGLpPCr6vX6VR1i1U/eyB25FWhpkWNAA8XzOrq5emNTCrBxFg/AEDWmTprhUdEJIoBJUCZmZl45plnEBIS0u24IAgoKTEu2ubi4oKHH3548BESXYSpB8jPzbo9QMC5mWCnqlqt/tn99WHXQop3p0Retlh8YowfZBIJSurbUdbQYY3wiIhEMaAEaNiwYait7VkoWV9fzwUSySp0hnMLEVp7CAwAgrxMPUC2nQAV17Xhx1O1kEjQp8UOPVUKJIV7AWAvEBE5tgElQBcr/GxtbYVKpRpUQER90dSuhQBAIZNctKZlKAV3FUKfqrLtIbCP9hl7f64eHohIP7c+nTM5zh8AcKy8icXQROSw+vWTIz09HQAgkUjwpz/9CW5u576h6vV67N27F+PGjbNogES9Mc0A83VzgUQisfrnn98DJAiCKDFcjsEg4PODZQD6t1dYpJ8bPJXGYujCmjYMD/YcqhCJiETTrwQoJycHgLEH6MiRI3BxOTf04OLigrFjx+Kpp56ybIREvThXAG394S8ACPBQQgKgqUOLmla1eWq8LckpbURNixqeSjmuHRnU5/OkEgkSQj2xv6gBxyuamQARkUPqVwK0c+dOAMCvf/1r/N///R+8vLyGJCiiyxFrCryJQiaFn7sL6to0OFXVapMJ0LbjlQCAaxOC+r1SdmKoF/YXNSCvohm3jQ2zyR4uIqLBGFAN0Hvvvcfkh0RVL+IMMJMgL2PSc9JG64Ayj1cBAGaMCu73ucMCPeAik6K5U4eyRs4GIyLH0+ceoNtvvx0bN26El5cXbr/99ku2/fzzzwcdGNGliD0EBgDBnkrkVQAnbXAq/OnqVpypaYOLTIrpIwL7fb5CJsXwYA8cK29GXkUzInz7VkBNRGQv+pwAeXt7m7vBvb29L9OaaGjVizwEBpzrAbLFmWCm4a8p8f7wVA1socgrQr26EqAW3HhFyOVPICKyI31OgN57771e/0xkbR0aPTq1xunZYg6BmTZFPVnVYnMzwbYd6xr+GkTiMjLEExIAlc2daOzqcSMichQDqgHq6OhAe3u7+e/FxcVYtWoVtm3bZrHAiC7GNPzloZT3u7jXkgI8lJBKgOZOnU3tDF/d0onc0kYAwA2JfZ/9dSE3FznCfV0BAIW1bZYIjYjIZgzop8fs2bOxadMmAEBjYyMmTpyIN998E7Nnz8batWstGiDRhc6tAWTdPcAupJBJEePvDsC2CqF/Pm1cpT0p3Ms8TDdQsQHG+2MCRESOZkAJ0MGDBzFt2jQAwGeffYaQkBAUFxdj06ZNWL16tUUDJLqQeQ8wEet/TIYHewCwrULon04Zt7C4Kr7/xc8XiutKgM4wASIiBzOgBKi9vR2ensbF0bZt24bbb78dUqkUkydPRnFxsUUDJLqQLcwAMxnRtUigrRRCC4Jg7gG6Kj5g0NeL9neHVGLsdSvndHgiciADSoDi4+OxdetWlJaW4rvvvsOMGTMAANXV1VwfiIZcw3nbYIjNtEqyrQyBFdS0obK5Ey5yKVJifAd9PZVChjAfYx3Q3kJujkpEjmNACdCf/vQnPPXUU4iJicGkSZOQmpoKwNgbNH78eIsGSHShhnbxdoG/0IiuIbBTVa0X3STYmky9PxNifKFSyCxyTdMw2C8F9Ra5HhGRLRhQAnTnnXeipKQEBw4cwLfffms+fv311+Pvf/97v661Zs0axMbGQqVSITk5GT/++OMl2+/evRvJyclQqVSIi4vDunXrur3/+eefIyUlBT4+PnB3d8e4cePw/vvv9ysmsl2CIJinZNtCD1BsgDtkUgla1DpUNneKHQ5+6kqApgwb/PCXSWyAMcn7hT1ARORABjyHOCQkBOPHj4dUeu4SEydOREJCQp+vsXnzZixevBhLly5FTk4Opk2bhpkzZ6KkpKTX9oWFhZg1axamTZuGnJwcPP/883jiiSewZcsWcxs/Pz8sXboUWVlZOHz4MH7961/j17/+Nb777ruB3irZkJpWNXQGARIA3q7izgIDAKVchhh/4yrJYhdC6/QG/HLGVABtuQQo2t8NUglQXNfOOiAichgDSoDa2trw4osvYsqUKYiPj0dcXFy3V1+tXLkSjz76KBYsWIDExESsWrUKkZGRF51Kv27dOkRFRWHVqlVITEzEggULMH/+fKxYscLc5pprrsHcuXORmJiIYcOG4cknn8SYMWPw008/DeRWycaU1ht/AHu7KiCT2sbCg7ZSCH2krAktnTp4qeRICrfcau3n1wHtK+QwGBE5hn7tBm+yYMEC7N69Gw8++CBCQ0MHtAKuRqNBdnY2nn322W7HZ8yYgT179vR6TlZWlrng2uSmm27C+vXrodVqoVB07xEQBAHff/898vPz8de//rXfMZLtOdtgXIDTxwaGv0yGB3vim6OVohdC7ykw9v6kDvO3eHIY7eeGsw0dyClpwJzx4Ra9NhGRGAaUAH3zzTf4+uuvMXXq1AF/cG1tLfR6PYKDu+9UHRwcjMrKyl7Pqays7LW9TqdDbW0tQkNDAQBNTU0IDw+HWq2GTCbDmjVrcOONN140FrVaDbX63Eq+zc3NA70tGmJnG4w9QGIvgni+ETayFtD+ImPvzMRYf4tfO9LPDSioQ07XCtNERPZuQAmQr68v/Pz8LBLAhb1Hl9tTqbf2Fx739PREbm4uWltbsWPHDqSnpyMuLg7XXHNNr9dcvnw5Xn755QHeAVmTOQGygRlgJqYhsNPVraLtCaY3CMguagAATIyxzL/N80X6Geucjpc3o1Ort9gMMyIisQyoBuiVV17Bn/70p277gfVXQEAAZDJZj96e6urqHr08JiEhIb22l8vl8Pc/91uvVCpFfHw8xo0bhz/84Q+48847sXz58ovG8txzz6Gpqcn8Ki0tHfB90dAyDYHZwgwwkxh/d8ilErSqdShvEmcm2InKZrSodXB3kSEx1NPi1/dxVSDQUwmdQcCx8iaLX5+IyNoG1AP05ptvoqCgAMHBwYiJielRe3Pw4MHLXsPFxQXJycnIzMzE3LlzzcczMzMxe/bsXs9JTU3FV1991e3Ytm3bkJKS0iOG8wmC0G2I60JKpRJKpfKyMZP4bHEIzEUuRVygO05WteJkVQvCuwqGrWl/V3HyldG+kMssv0GsRCLB+EgfbDtehZySRiRHW76XiYjImgaUAM2ZM8ciH56eno4HH3wQKSkpSE1NxTvvvIOSkhKkpaUBMPbMlJWVmTdeTUtLwz//+U+kp6dj4cKFyMrKwvr16/Hxxx+br7l8+XKkpKRg2LBh0Gg0yMjIwKZNm7hJqwMwGASUmRMg2+kBAoyF0CerWnGqqgXXjhz4DuwDtX8Ih79MxkWdS4CIiOzdgBKgl156ySIfPm/ePNTV1WHZsmWoqKhAUlISMjIyEB0dDQCoqKjotiZQbGwsMjIysGTJErz11lsICwvD6tWrcccdd5jbtLW1YdGiRTh79ixcXV2RkJCADz74APPmzbNIzCSemlY1NHoDpBLAywbWADrfiCBPfI0KUQqhBUHAvq4C6AmxQ5cAjY80bq2RU9IwZJ9BRGQtA0qAAKCxsRGfffYZCgoK8Mc//hF+fn44ePAggoODER7e92myixYtwqJFi3p9b+PGjT2OTZ8+/ZJDbK+++ipeffXVPn8+2Q9T/Y+XDa0BZHJuSwzrT4Wvb9OgpkUNhUyCcZE+Q/Y5YyK8IZUA5U2dqGzqRIi3asg+i4hoqA0oATp8+DBuuOEGeHt7o6ioCAsXLoSfnx+++OILFBcXm4esiCzprI0OfwHnNkU9Vd0Kg0GA1IoJWlFdGwBgTITPkM7OclfKMTLEC3kVzcgtbcCvvEOH7LOG2kd7e19t/nz3TYqyQiREJJYBJUDp6el45JFH8Le//Q2enudmnMycORP33XefxYIjOl9pve3NADOJ8XeDi0yKdo0eZY0d5mnj1lBUa3wuEy5R/9OXH/h9MT7KB3kVzcgpacSvkuw3ASIiGtB0kf379+M3v/lNj+Ph4eEXXcSQaLBscQaYiVxmnAkGwOorQpt6gCbG+g75Z42L8AFg3HaDiMieDSgBUqlUva6WnJ+fj8DAwEEHRdQbWx4CA84Ng1mzELqlU4u6Ng0kElhlavqocC8AwNGyJvMipERE9mhACdDs2bOxbNkyaLVaAMY1QkpKSvDss892m5FFZEnmfcDcba8HCABGBFm/ELqozvhMRgZ7wtsKM+OGB3nCRSZFc6fOnJASEdmjASVAK1asQE1NDYKCgtDR0YHp06cjPj4enp6eeO211ywdI5FxDaBG4w9cP1vvAaq2YgJUaxr+ss7ChC5yKUaGGO+Tw2BEZM8GVATt5eWFn376CTt37kR2djYMBgOuvPJK3HDDDZaOjwgAUNXSCa1egFwqgafKNnuATInB6epW6A2CVabqm+p/LlUAbWlJ4V44UtaEo2VNmDWahdBEZJ/6nQAZDAZs3LgRn3/+OYqKiiCRSBAbG4uQkBDRNoIkx2cabgn1UdncGkAmUX5ucHORoV2jR2FtK+KDLL8n1/k6tXpUdu09Zq0eIAAYFeYNoBRHy3vWARIR2Yt+DYEJgoDbbrsNCxYsQFlZGUaPHo1Ro0ahuLgYjzzySLc9vYgsyVT/E+Fjvenl/SWTSpDQ1Qt0zArJQXFdOwQAfu4uCPay3qKEo8O9AQDHWAhNRHasXz1AGzduxA8//IAdO3bg2muv7fbe999/jzlz5mDTpk146KGHLBok0dl6Yw9QpJ/1NxrtjyvCvHCwpBHHy5sxe1zfV0QfCNPwV4y/dZPCkSGekEklqGvToLK5E6Hetv3/hIioN/3qAfr444/x/PPP90h+AOC6667Ds88+iw8//NBiwRGZlJp6gHxttwcIMA0PAccrhr4H6FwC5D7kn3U+lUKG4V0z3o6cZSE0EdmnfiVAhw8fxq9+9auLvj9z5kwcOnRo0EERXchUAxTha9u9DVeEGtfJOVbePKTDQ1q9wfxMrJ0AAUBS1zAY64CIyF71KwGqr69HcHDwRd8PDg5GQwN3iibLO5cA2XYPkGl4qL5Ng6pm9ZB9ztmGDugNAtyVcvh7WH9ZgKSwrkSPU+GJyE71KwHS6/WQyy9eNiSTyaDT6QYdFNH59AYB5Y320QOkUsgwrGtLjGPlQ5ccFJ9X/yPGzMtzPUBMgIjIPvWrCFoQBDzyyCNQKpW9vq9WD91vvOS8qpo7oTMIUMgkVp3tNFCjwrxxsqoVx8ubcX3ixXtMB0Os+h+TK8K8IJEAVc1qVLd0IsjT9v+/EBGdr18J0MMPP3zZNpwBRpZm2gU+zMfVZtcAOt8VoV74IqdsyKbCGwQBxV1bYMQEGBMgS+323lduLnIMC/TA6epWHCtrRlACEyAisi/9SoDee++9oYqD6KLspQDaZFRXfcxQzQSrbOqEWmeAUi5FqLd4iUdSmBdOV7fiaFkTrk0IEi0OIqKBGNBeYETWZE6AbHgRxPMlds0EK6lvR3On1uLXNw1/Rfm5QSriyuusAyIie8YEiGyeaRVoW18E0cTX3QXhPsZYjw7BLCnTBqim4S+xmNY8OlrGqfBEZH+YAJHNs5cp8OcbG2lMDnJLGy16XUEQUGSq/xGpANpkVLixp6ussQMNbRpRYyEi6i8mQGTzzq0CbR89QAAwPtIXAJBT0mjR69a1adCq1kEmlYj+PLxUCvM2HBwGIyJ7wwSIbJpOb0BF147n9tQDND7KB4AxAbLkitBnaozDX5G+rlDIxP/nOyqcw2BEZJ/E/w5KdAmVzZ3QGwS4yKQI8ux9/SlblBTuDblUgtpWtXkIzxIKaloBAMMCPSx2zcFICmMhNBHZJyZAZNNKu3aBD/NRQWoHawCZqBQyXNE1HT7HQnVABkHAma4EKM5WEqBwbolBRPaJCRDZtFLzDDD7Gf4yGR/pAwDItVAdUFVzJ9o0eihkEpuZEWeaCVZUNzRT/omIhgoTILJpplWg7TIBiuoqhC61zAbBpvqfGH93yKW28U/X77wp/8e5MzwR2RHb+C5KdBHmBMiOCqBNruxKgI6VNUOt0w/6erZW/2NiWvl6KNY8IiIaKkyAyKaVdhUQR9lhD1Cknyv83V2g0RsG3TuiNwgo7FoA0dYSINOK0EO19xkR0VBgAkQ27dwQmG3UvPSHRCIxT4fPLh7cMFh5YwfUOgNcFTKE+tjWxqOmQmj2ABGRPWECRDarU6tHdYsagH0OgQHAhBg/AMCegrpBXedUtXH4KzbAXdT9v3pjKoQuqGlFh2bwQ31ERNbABIhslmkPMA+lHD5uCpGjGZhpwwMBAL+cqYNGZxjwdU5UGoeXRgZ7WiQuSwryVCLAQwmDAORVchiMiOwDEyCyWaY1gCJ8XSGxsV6PvkoI8USAhwvaNXocLBnYMFhzp9a8mOLIUNtLgCQSCdcDclIdGj3UWvb6kX2Six0A0cWY1gCyxwJoE6lUgqnxAfhvbjl+OlWLyXH+/b5GfmULAGMi6KWyzZ6wpDBv7Mqv4ZYYTuBsQzt25degrLEDTR1aSAAEeCpxtLwJv50+zC6XrCDnxB4gsln2vAbQ+UzDYD+eqhnQ+XkVxqQiIcTLYjFZmrkHqII9QI6qVa3Dfw6UYs2uAhyvaEZTh3HhSwFATYsaH+0twfUrd2PFd/noZK8Q2QH2AJHNMg2BRdrRLvC9uSo+AABwuKwJje0a+Li59PncDo0ep7sKoBNtcPjLxFQInV/ZAo3OABc5f7dyJOWNHXjnhwLUtmoAGFc5nxjrh2AvFXQGAaX17Thd3YqsM3X4587T+LmgFhsengBf975/rRNZG79Lkc0qcZAeoBBvFUYEe0AQgJ9P92822M+na6EzCPBxUyDEy7amv58vwtcV3q4KaPUCTla1iB0OWVBxXRvuWpeF2lYNfFwV+O30YbgrJRLR/u5QKWTwUMqRGOqFjxZOwroHroS3qwI5JY24c90elDVabiNgIktjAkQ2yxFqgEyuih/YMFjm8SoAxuEvWy4El0gk5hWhj3FneIfR1K7FA+v3oqyxA/7uLnjs6riL/kIikUjwq6RQfJaWilBvFQpq2nDfu7+gvk1j5aiJ+oZDYGSTmtq1aOnUAQAi7HQNoPNdMzIQG34uxLbjVVg2u29DRB0aPTKOVgA4t92ELUsK98aegjquCG1lH+0tuWyb+yZF9fu6BoOA9E9zUVrfgSg/N9w/KQqefSjCHx7siS2/nYJ572ShuK4dae9n4/0FE6GUy/odA9FQYg8Q2SRT70+AhxKuLvb/jXPKMH8EeipR36bBzvzqPp2TcaQCLZ06+LopEBvgPsQRDh73BHMsb/9wBjtOVMNFLsWa+6/sU/JjEubjig0PT4CnUo59RfV4/vOjEARhCKMl6j8mQGST7HkLjN7IZVLcPj4cALAl+2yfztm8vxQAkBLjZ3OrP/fGtCfY8Ypm6A38YWfPjpY1YcW2fADAsttGmf/f9sfwYE+8df+VkEkl2HLwLLYcLLN0mESDwgSIbFKJHe8CfzF3JEcAAL4/UY26VvUl256ubsW+onpIJUBy167yti7W3x3uLjJ0ag0407VzPdkfg0HAC1uPQm8QcPPoUMybEDnga109IhDpN44AALz036MoqWu3VJhEg8YEiGySIxVAm4wI9sSYCG/oDAL+m1t+ybafHjD2/lyXEAQvV9tc/PBCUqkEiaFdw2AshLZbH+8vQW5pIzyUcvzp1isGXXyfNn0YJsb4oU2jx5JPc6HTD3xLGCJLYhE02STzGkAOMgRmcmdyBA6fbcJ/ss/i11Njev3h0typxWddw2TzJkShpuXSvUW2JCncGweKG3CsrBlzx4sdDfVXbasaf/3mBADgDzNGINgCSy/IpBK8efdYzPq/H5Fd3ID3fi7CwqvjBny9oSr6JufDHiCySaYeIEcaAgOAW8eEQSmXIq+i+aK9QCu3nUR9mwZxAe64dmSglSMcHHMhNHuA7NI/dpxCc6cOo8K88ODkaItdN9LPDS/ckggA+Pv2k1wfiGwCEyCyOQaDgLPmHiDHSoB83V3wxPXDAQCv/O84Gtu7r5FytKwJm7KKAADLZidBLrOvf6KmYtljZc0wsBDarpxtaMdH+4y9K0tnJVr8a++u5EhMiPFFu0aPl788ZtFrEw2EfX13JadQ3aKGRm+ATCpBqLftrn48UAunxWF4kAfq2jRYnnHCfFyjM+CFrUdhEIBbx4bhquEBIkY5MPFBHnCRS9Gi1pl78cg+rN5xClq9gCnD/DEl3vJfe1KpBK/NHQ25VIJtx6vMi3wSiYUJENkc0w/OMB+V3fWA9IWLXIrXbx8NANh8oBRp72fji5yzmPl/P5iLT1+4OVHkKAdGIZMiIcS4ZxkXRLQfBTWt5rqzp24aOWSfMyLY01z/89rXx6HRsSCaxON4P13I7pU64BT4C02I8cMT18UDAL49Voklmw+hoKYNAR4u+Me94y1SfCoW08aoXBDRfvxjxykYBOD6hCBcOcTLLjx+bTwCPJQoqmvHB78UD+lnEV0KEyCyOed2gXfcBAgA0meMxLYlV2POuDC4u8jwwOQo7Ei/BtcmBIkd2qAkhRsLoY8wAbILZxva8dVh45Yri28YMeSf56GUm9cGWv39KTS1a4f8M4l6I3oCtGbNGsTGxkKlUiE5ORk//vjjJdvv3r0bycnJUKlUiIuLw7p167q9/+6772LatGnw9fWFr68vbrjhBuzbt28ob4EsrMTBVoG+lBHBnlh1z3gcffkmvDpnNLzd7GPNn0sZG+EDADhU2shCaDuw4aci6A3G2p/REf1f8Xkg7k6JwIhgDzS2a/HPnaes8plEFxI1Adq8eTMWL16MpUuXIicnB9OmTcPMmTNRUtL7Og+FhYWYNWsWpk2bhpycHDz//PN44oknsGXLFnObXbt24d5778XOnTuRlZWFqKgozJgxA2VlXIbdXpinwDvYDLBLseWd3vtrZIgnlHIpmjt1KKprEzscuoSmdi0+2W/8fvvYINbm6S+5TIrnZhnr3P6dVYzKpk6rfTaRiagJ0MqVK/Hoo49iwYIFSExMxKpVqxAZGYm1a9f22n7dunWIiorCqlWrkJiYiAULFmD+/PlYsWKFuc2HH36IRYsWYdy4cUhISMC7774Lg8GAHTt2WOu2aJDO1jtfAuRIFDKpeTr8obON4gZDl/TB3mK0a/RICPHE9BHWXXPqmhGBmBDjC43OgLW7Tlv1s4kAEVeC1mg0yM7OxrPPPtvt+IwZM7Bnz55ez8nKysKMGTO6Hbvpppuwfv16aLVaKBQ9hw/a29uh1Wrh5+dnueBpyGh0BlQ0G38bdPQaIHvTnxV4x0b4ILu4AYdKmzB3fMRQh0YDoNUb8O89RQCMSzNYuxdSIpFgyQ0jcN+/9uLjfaX4zfRhCPNx/GFvsh2i9QDV1tZCr9cjODi42/Hg4GBUVlb2ek5lZWWv7XU6HWpra3s959lnn0V4eDhuuOGGi8aiVqvR3Nzc7UXiKG/sgCAArgoZAjxcxA6HBmhclA8AILe0UdQ46OK2HatCdYsaAR5K3Do2TJQYUof5Y1KsHzR6A9awF4isTPQi6At/6xAE4ZK/ifTWvrfjAPC3v/0NH3/8MT7//HOoVBefVrx8+XJ4e3ubX5GRA9/9mAbn/AJoR6qLcTbjugqhj5c3c60XG/X+L0UAgHsnRsJFLs6PAolEgiVdM8I27y9FRRO3yCDrES0BCggIgEwm69HbU11d3aOXxyQkJKTX9nK5HP7+/t2Or1ixAq+//jq2bduGMWPGXDKW5557Dk1NTeZXaWnpAO6ILMFR9wBzNpF+rvB1U0CjN+BEJXtUbc2pqhb8cqYeUglw70RxNw6dHGfsBdLqBaz/sVDUWMi5iJYAubi4IDk5GZmZmd2OZ2ZmYsqUKb2ek5qa2qP9tm3bkJKS0q3+54033sArr7yCb7/9FikpKZeNRalUwsvLq9uLxFHqoHuAORuJRIKxkT4AOAxmi0wLEN6QGGwTdTdp1wwDAHy8r4TrApHViDoElp6ejn/961/YsGED8vLysGTJEpSUlCAtLQ2AsWfmoYceMrdPS0tDcXEx0tPTkZeXhw0bNmD9+vV46qmnzG3+9re/4YUXXsCGDRsQExODyspKVFZWorW11er3R/1n6gGK8BX/mzINjmk9ICZAtqVNrcOWg8ZlQR5MtdyO74NxzYhAJIR4ok2jxwd7uTo0WYeoCdC8efOwatUqLFu2DOPGjcMPP/yAjIwMREcb/1FWVFR0WxMoNjYWGRkZ2LVrF8aNG4dXXnkFq1evxh133GFus2bNGmg0Gtx5550IDQ01v86fKk+2q6TOmABFsQfI7o1jD5BN+vpIBVrVOsT4u2HqMNvYcFcikeA3043rEL33cyE6tXqRIyJnINo0eJNFixZh0aJFvb63cePGHsemT5+OgwcPXvR6RUVFFoqMrE0QBPPCebEB7iJHQ4NlSoDO1LShoU0DX3fO6rMF/zlgrHG8KyUSUqntTDS4ZUwY3vg2H+VNnfgip0z02iRyfKLPAiMyaWjXoqVTB4mENUCOwNfdBfFBHgCA7OIGkaMhADhT04r9RQ2QSoA7rrSt9ZkUMil+PTUWAPDvPUXmGb5EQ4UJENmMwlpj70+olwoqhUzkaMgSJsQYdxbfX1wvciQEAJ9lnwUATB8RiBDviy8NIpa7UyLhqpDhRGUL9hbya4aGFhMgshnFXcNf0f4c/nIUKdHGFdj384eZ6HR6A7YcNCZAd6fY5lpn3m4KzBkfDgDYlFUkbjDk8ESvASIyKeoqgI4JsJ/hr8ttD2HaGsJZTYgxJkBHyprQqdWzZ09EP56qRVWzGn7uLrg+sfe11mzBw1Oi8fG+Enx3rArljR02MU2fHBMTILIZ7AFyPJF+rgjyVKK6RY1DpY0oqLn87vDOnjQOlS9yjFPfbxsbJtrKz32REOKFSbF+2FtYjw/3FuOPNyWIHRI5KNv9V0BOx9wD5G8/PUB0aRKJxNwLdICF0KJpU+uQebwKADC3a4jJlj08JQYA8OmBs9DquZUKDQ0mQGQzTD1AMZwC71BSTIXQRawDEsv2vCp0aPWI8XfDmAhvscO5rBsSgxHg4YKaFjW+P1EtdjjkoJgAkU1obNegsWsJfC6C6FhMPUDZxQ0wcGqzKP6bWw4AuG1cuF1sMuwil5qn6W/ez70ZaWgwASKbUNw1/BXspYSbC0vTHElCiCfcXWRo6dShsqlT7HCcTptahx9O1gAw1v/Yi3kTjDPVduVXc5d4GhJMgMgmFLEA2mHJZVJMjDX2Ap2u5p581nakrAk6g4CkcC/zwpT2IC7QAxNj/WAQgP8cOCt2OOSAmACRTSiqZQG0I5s2PBAAcLrGNhKgNrUO2cX12J5Xha05ZfjlTB1a1TqxwxoSh842AgBmj7X94ucL3TvR2Au0eX8pDAYOn5JlcayBbAKnwDu2acONm24W1bZBqzdAIRPnd68OjR4bfi7E6h2noNZ1n1301aFyJIV7Y864cLi6OMZ6RQ3tGhTXtUMiAW61o+Evk5lJofjT1mMoa+zAvqJ6TI7zFzskciDsASKbYBoCi2EC5JDigzwQ7KWEznBuw1trq2jqwK3//AlvfJcPtc6AYC8lJsT4YfqIQIT7uEKAcbjorV2nHabm5PDZJgDA5Fh/m9z64nJUChluHhMKAPj8IIfByLLYA0Q2wVQEHc0hMIckkUgwbXggPss+i9PVrRge5GnVzz9T04oH1+9DWWMHgjyVmD4iEGMjfSDtmhF10yjgbEM7PtpXgvo2DdbtLsB1CUEYH+Vr1Tgt7VBpIwBg9rhL9/5cbkXzvujLNQayyOXc8eH4ZH8pMo5U4uXbkgYSGlGv2ANEomtq16KuTQOAawA5MtMwmLULoUvr23H321koa+xAbIA7Pl80BeOjfM3Jj0mErxt+d0084gLcodULeOz9bJQ32m9PUGVzJyqbOyGTSDAzKVTscAZsQowfInxd0arWITOvSuxwyIEwASLRFdQafyAGeynhoWSnpKOaGm9MgCqaOtHSqbXKZ7ZrdHjs/WzUtmqQEOKJ/6SlIsL34r2Mbko5HpwcjRAvFWpa1Fi46QDaNfZZHH24q/dnRIgnvN0U4gYzCFKpBLd3rV7NYTCyJP60IdGd6dofKi7AfqboUv8FeCgR6q1CRVMnCmpaMS5yaIeXBEHA058dRl5FMwI8XLDhkQkI8FBe9jylQoYHJ0djw8+FOFbejFe/zsPrc0cPaayWJgiCefbX2AhviwxxWcJANw+ee2UEVn9/Gj+crEFqnD88Vfab0JHtYA8Qie5M19TouEAOfzm6EcHG2p/jFS1D/lnv/1KM/x2ugFwqwZr7k/u1q7ivuwv+ce94AMYf2vsK7Wsbj5L6djS0a+EilyIhxEvscAYtNsAd4yJ9YBCAo2VNYodDDoI9QCQ6Uw/QsEBjD5Ct/LZKljcqzAu7T9Ygv7IZGp1hyHYlL61vx1++OQEAeH5Wonkhxv6YEh+AeyZE4pP9pXju88P4+olpUCnsY3q8qfdnVKiXTe/83h+3jAlFbmkjDpc1IXVYgNjhkANwjH8ZZNcK2APkNMJ9XOHjpoBWL+Bk1dD0AgmCgGe2HEa7Ro+JsX54pGtn8YF4bmYiAj2VKKhpw5pdBZYLcgjpDQKOdE1/HxvpI24wFmSaDl9c146mDuvUkJFjYwJEotIbBPMUeFMPEDkuiUSCpDDjbuRHy4dmKOOT/aXYU1AHlUKKv90xBlLpwDf/9HZT4OXbRgEA3t5dYBd7mRXUtKJNo4e7i8yh/k2FertiQoyxbuwIh8HIApgAkajONrRDozcOhfSnRoPsV1K4MQE6UdkCrd5wmdb9U97Ygde+zgMAPDVjpEWWVZiZFIIJMb5Q6wxYtf3koK831Exr/4yO8IZsEMmfLbp5tLEX6EjXEB/RYDABIlGZ6n9i/d0d7ps19S7C1xXergpodAaLrgkkCAKe/+IIWtU6jI/ywa+nxlrkuhKJBM/OTAAAfHqgFKeGaOjOEjQ6A45VNAMAxkb4iBvMEJg1OhQSAKUNHWho14gdDtk5JkAkKlP9z7Ag1v84C6lEglFhxplJhy34m/yWg2XYlV8DF7kUb9w5xqIJdXK0H2ZcEQyDAPz123yLXdfS8rqKy33dFIjyc7xV1YO8VOZePVOdE9FAcRYYiaqAawA5pbERPthTUIdj5c1oVesGvQBmdXMnln11DACw+IbhiB+CrTae/lUCtudVYXteFY6WNZmH8mxJbkkjAGPxs0Rifz2qfZkBOibCG4W1bThS1oSrRwRaISpyVOwBIlFxDSDnFOHrinAfV+gMAg4UDW6NHUEQsHTrUTR36jA63BuPTYuzUJTdxQd54LauHdXX7Do9JJ8xGK1qHU5VG4fnxjnQ7K8LjQrzhlQClDV2oK5VLXY4ZMeYAJGoztR29QA50GwVujyJRILUOH8AwN7CeugNwoCv9dXhCmQer4JCJsEbd42BXDZ039Z+e008AOCbo5Xm4VtbceRsIwyCcamBIE/72/m9rzyUcvP3C84Go8FgAkSiaenUoqbF+Bsce4Ccz+gIb7i5yNDUoUVeV+Fuf9W0qPHnL41DX49fGz/kqx6PDPHEDYlBEATjtHhbktM1+8uRe39MRncNPzIBosFgDRCJxjQDLMBDCS/u7eMQ+rOKt0ImxYQYP+w+WYOsM3X9rqkxGAQ89Z9DqG8zbnS6qKt3ZqgtujYe2/Oq8UVOGRbfMMImlm+obVHjbEMHpBJjjYyjGxXmhf/mlqGiqRM1LWoEel5+jzeiC7EHiERjWgl4eBCHv5zVpFg/SCVAYW0b8iv7N718454i7D5ZA6Vciv+7Z7zVtny4MsoXk+P8oNULePfHM1b5zMvJ7ZpNFx/k4RQbhbq5yBHf9X3jcFmjuMGQ3WIPEInmVNcaMCOCHTcB6kuPyMV2wHYGPm4umDIsAD+drsVXh8sRFzi8T+cdK28y7/W19OZEjAyx/KyvS1l0TTx+ObMPn+wrxe+vGw4/dxerfv75BEFArnn4y1e0OKxtTLgPTla14sjZJlyfECx2OGSH2ANEojH9xj/Cyj+8yLZcnxAEL5Uc9W0a/Hiq9rLtyxo7MH/jfmj0BlyfEIQHJ0dbIcrupg0PQFK4Fzq0emz8udDqn3++kvp21Ldp4CKT4opQ+9/5va8SQ70gk0hQ3aJGdYvtb1FCtocJEInGtKLuiGAmQM5MqZBhZtcWB7vyq3H0EoWtje0aPLxhH6qa1RgR7IGVd48TZb0biUSCx7tqjjbuKUKrWmf1GExMvT+jwhxn5/e+cHWRmRdQPVY+sCJ6cm7O86+FbEpLpxblXRtLjhiCRevIvowJ90Z8kAd0BgH3vfuLeT+r852qasG8t3/B6epWhHipsPHXE+HtJl69y02jQhAX6I7mTh0+/KVYlBg0OgMOd62IPC7KR5QYxGTaWPcYZ4PRALAGiERhqv8J8lSK+kOMbINEIsF9E6OwcU8RSurb8cC/9uJ318Vj1uhQtHTqsOtkNVbvOIVOrQEBHi7YOH+C6LOvpFIJ0qYPw9OfHca/firEw1NioFLIurW5XA3YYOu/duVXo0Orh6dS7lA7v/dVYqgXpLllKG/qRF2rGv4enA1GfccEiERxsqv+x9rFq2S7VAoZfj0lBt8crcS+onos/+YElncVOptMGx6AN+8ee9mF/vozHX8w5owLx98zT6KiqROfHyyzekH7FzllAIxbX0jtcOuLwXJXyhEX4IHTNa04Vt7MrTGoXzgERqI4WWXsARrO4S86j1Ihw6ZHJ+LVOUmYHOcHiQRwd5Fh2vAAvDY3Cf/+9USbWuXYRS7Fgq6tN97+oWBQK1r3V02LGpnHqwAA451w+MtkVLix8PtoOYfBqH/YA0SiMO1Z5MhT4GlgVAoZHpgcjQcmR6NNrYNSLh3S7S0G696JkfjH96dQXNeOjCMVuLVrv7Ch9ln2WegMAiJ8XRHqLf5ijGK5ItQLX+aW42xDBxraNfB1E29JArIvTIBIFKZFEDkFni7FfZC7xA9GX+t33FzkeGRKDFZtP4W1uwpwy5jQIZ+ZZjAI+HifMb6JMX5D+lm2zlOlQEyAOwpr23CsrAlXDecwGPWN7f5aRQ6rqV2LqmbjHmBcBZocwcOpMXBzkeF4RTN2n6wZ8s/7uaAWJfXt8FTKMSbCZ8g/z9aZtlE5yunw1A9MgMjqTnYNf4V5q5xi2X5yfL7uLrh3orFHaO2uod8k1dT7M2d8uFOt/XMxo0K9IIFxUcimDq3Y4ZCd4BAYWZ15DzAugAjAejOWaGgtmBaLTVlF2FtYj+ziBiRHD822FJVNndh2zFj8fN+kKOSUNA7J59gTL1cFovzcUFzfjmMshqY+4q8OZHUnKjgFnhxPqLcrbh8fAWBoe4He21MInUHAxBg/JDrR1heXYx4GK+MwGPUNEyCyuuMVxm9Qo8L4zZscy2PT4yCRANvzqi65pcdAtXRq8dEvxh7Dx66Os/j17Znp+0lxXRv3BqM+YQJEVqU3CMjrSoCcaeNGcg7DAj1wW9c0+BXb8i1+/c37S9Gi1mFYoDuuSwiy+PXtmY+bCyJ9XSEA+K5riJDoUpgAkVUV17WhXaOHSiFFnBMu3U+OL/3GEZBLJdiVX4PC2jaLXVerN2DDT8ad5xdOi4NU6nwrP1+OaRjsmyMVIkdC9oAJEFmVadfmkSFekPEbODmgaH93zJsQCQD47lglBMEyq0N/cdC451WAhxJzxodb5JqOZlTX5qh7C+tR16oWORqydUyAyKpY/0PO4Inrh0OlkKKkvt38NT8YHRo9VmaeBACkTY/rsekqGfm5uyDcxxV6g2DeJoToYpgAkVWZeoBY/0OOLNhLhQVXGYuU/3e4AmqdflDXe29PISqbOxHu44oHU6MtEaLDMv1ylXG0UuRIyNaJngCtWbMGsbGxUKlUSE5Oxo8//njJ9rt370ZycjJUKhXi4uKwbt26bu8fO3YMd9xxB2JiYiCRSLBq1aohjJ76QxAEHO9ao4M9QOToHr82Hr5uCjR1aPH9ieoBX6ehTWOeVv+HGSOglLP351KSuobB9pyuRWO7RuRoyJaJmgBt3rwZixcvxtKlS5GTk4Np06Zh5syZKCnpfWG4wsJCzJo1C9OmTUNOTg6ef/55PPHEE9iyZYu5TXt7O+Li4vCXv/wFISEh1roV6oOaFjVqWzWQSoCEECZA5NhcXWTmGWE/n65FRVPHgK7zxrZ8tHTqkBjqhTnjWPtzOQGeSiSEeEJnELCNw2B0CaImQCtXrsSjjz6KBQsWIDExEatWrUJkZCTWrl3ba/t169YhKioKq1atQmJiIhYsWID58+djxYoV5jYTJkzAG2+8gXvuuQdKpdJat0J9YBr+igv0gKsLf4slxzcyxAujwrxgEID/HDgLjc7Qr/O/P1FlXin8xVsSOfOrj2aNDgUAZHA2GF2CaAmQRqNBdnY2ZsyY0e34jBkzsGfPnl7PycrK6tH+pptuwoEDB6DVcv8XW8cCaHJGt44Ng4dSjsrmTmzNLevzrLC6VjWe/uwIAGD+1FhMGRYwlGE6lFvGGBOgn07Vor6Nw2DUO9H2AqutrYVer0dwcHC348HBwais7L14rbKystf2Op0OtbW1CA0NHVAsarUaavW5KZPNzVxKfSiY9uhhATQNhq3sndbXOLxUCtw7MQrrfzqD3NJGRPi6XjaZUev0WLw5F7WtaowI9sDTvxppiZCdRlygB5LCvXC0rBnfHK3A/ZNYOE49iV4ELZF079IVBKHHscu17+14fyxfvhze3t7mV2Rk5ICvRRd3+KypANpb5EiIrCs2wB2/SjL+gva/wxX44WTNRXuCOrV6PLYpGz+eqoVSLsXf543jtPcBuHWMsf7qq0PlIkdCtkq0BCggIAAymaxHb091dXWPXh6TkJCQXtvL5XL4+/sPOJbnnnsOTU1N5ldpaemAr0W9q2lR42xDByQSYEwkEyByPlOH+eOqeGPPz7fHKrE1t7zHYn35lS14eMM+7D5ZA5VCig2PTOAvDAN0c9cw2N7CelQ1c28w6km0ITAXFxckJycjMzMTc+fONR/PzMzE7Nmzez0nNTUVX331Vbdj27ZtQ0pKChQKxYBjUSqVLJgeYrmljQCA+EAPeKkG/v+KyF5JJBLMGh0KL1cFvjlSgf1F9Uj9y/eYmRQCT5UclU2d2HGiGoIAuLnIsOGRCZgcN/Bf7JxdhK8bkqN9kV3cgK8PV2D+VbFih0Q2RrQECADS09Px4IMPIiUlBampqXjnnXdQUlKCtLQ0AMaembKyMmzatAkAkJaWhn/+859IT0/HwoULkZWVhfXr1+Pjjz82X1Oj0eD48ePmP5eVlSE3NxceHh6Ij4+3/k0SACC3tAEA4OWqsJkaDiIxXBUfgEAPJbbnVaGssQP/ze0+RDNrdAjSbxyB+CBPkSJ0HLeOCUV2cQP+e6icCRD1IGoCNG/ePNTV1WHZsmWoqKhAUlISMjIyEB1tLFirqKjotiZQbGwsMjIysGTJErz11lsICwvD6tWrcccdd5jblJeXY/z48ea/r1ixAitWrMD06dOxa9cuq90bdWfqAYr0dRM3ECIbMDLEEyOCPTAixBM/nqyBRCKBSiHDNSMDkchJAhZz85gwvPJ1Hg6VNuJMTSs3YKZuRE2AAGDRokVYtGhRr+9t3Lixx7Hp06fj4MGDF71eTEyMxTYfJMvQGwQcKjUWQEf6uYocDZFtkEgkmBDjhwkxfmKH4rACPZW4Kj4Au0/WYGtOGdJncDYdnSP6LDByfAU1rWhV6+DmIkOQp0rscIjIidx+pXH17C/6sQYTOQcmQDTkcksaAQCjw70h40q2RGRFM64IgbuLDKX1HThQ3CB2OGRDmADRkMvpqv8ZF+UjahxE5HxcXWTmNZg+P1gmcjRkS5gA0ZDLKTH+1jU+0lfkSIjIGZmGwb4+XI5OrV7kaMhWMAGiIdWq1uFkVQsAYDx7gIhIBJPj/BHmrUJzp447xJOZ6LPAyLEdKKqHQQCi/NwQ7MUCaLo8rhNFliaTSnBXSiT+b8cpbN5fgtvGhokdEtkA9gDRkPrlTD0AYHIcp/oSkXjuSomARAL8fLoOJXXtYodDNoAJEA2pX87UAQCX9CciUUX4upn3Yvv0APd7JCZANIRa1TocKTMugDiJCRARieyeCVEAgP9kl0KnN4gcDYmNCRANmQNF9dAbBET5uSHchytAE5G4brgiCH7uLqhqVmNnfo3Y4ZDIWARNQ4b1P0SDx6Jwy1HKZbgrOQJv/3AGm7KKcOMVwWKHRCJiAkQWdf43668PG3e5FgR+Eyci2/DA5Gi88+MZ/HiqFqerWxEfxA1SnRWHwGhIqLV6lDV2AABiA9xFjoaIyCjSzw3XJxh7ft7PKhI3GBIVEyAaEkV1bTAIgK+bAj5uLmKHQ0Rk9siUGADAZ9ln0dKpFTcYEg0TIBoS+V2rPw8P8hQ5EiKi7qbG+2NYoDvaNHp8ln1W7HBIJEyAyOIEQUB+pTEBGhnCBIiIbItEIjH3Aq3/qZBT4p0Ui6DJ4qpb1Gho10IulWBYIAsMici6Ljfp4r5JUbgzORKrtp/C2YYOfHW4HHPHR1gpOrIV7AEiizP1/sQFusNFzi8xIrI9ri4yzL8qFgCwdlcBDAZB5IjI2vjTiSzOVP8zMpjDX0Rkux6YHA0PpRwnq1rx/YlqscMhK2MCRBbVodGjuK4NADAyxEvkaIiILs7bVYH7Jxu3x/jnztMQBPYCORPWAJFFnapugUEAAj2V8HPn9HeiS+ECoeJ79KpY/HtPEXJLG7Ejrxo3cHVop8EeILKo4xXNAIAEDn8RkR0I8lThkSnGWqAV2/JZC+REmACRxbRrdMjrSoCSwr1FjoaIqG9+O30YvFRynKhswZeHysUOh6yECRBZzPa8amj1AvzcXRDhy93ficg+eLsp8JvpwwAAKzNPQq3TixwRWQMTILKYL3ONvzmNifCGRCIRORoior779dQYBHoqUVLfjvU/FYodDlkBEyCyiKZ2LXafNE4jHRvhI24wRET95OYix/OzEgAA/9hxGuVdmzmT42ICRBbx7bEKaPUCQrxUCPZSiR0OEVG/zRkXjokxfujQ6vHq18fFDoeGGBMgsoj/njf8RURkjyQSCV6ePQoyqQQZRyqxk4sjOjSuA0SDdqamFXsK6iCRAGMjfcQOh4joki63/tLkWD/8XFCHJz7JwZPXDYebsuePyvsmRQ1VeGQl7AGiQXv/l2IAwHUjg+DrxsUPici+3XhFCAI9lGjp1GHroXKuEO2gmADRoLRrdPgs+ywA4MHUaJGjISIaPBe5FHelREAqAY6WNSG3tFHskGgIMAGiQdmaU46WTh1i/N1w9fBAscMhIrKICF83XJcQBADYmlvGWWEOiAkQDZggCObhrwcmR0Mq5do/ROQ4rhkZhBHBHtDqBXzwSzFa1TqxQyILYgJEA/bDqVrkVTRDpZDiruRIscMhIrIoqUSCeSlR8Hd3QWOHFh/tLYZGZxA7LLIQJkA0IIIgYGXmSQDAA5Oi4e2mEDkiIiLLc3WR4YHJ0VDKpSiqa8dH+4qh0zMJcgRMgGhAvj9RjUOljXBVyJB2zTCxwyEiGjLBXio8nBoDhUyCk1Wt+GR/KfcLcwBMgKjfzu/9eXhKDAI8lCJHREQ0tGIC3PHg5BjIpRIcr2jGwxv2oaldK3ZYNAhMgKjf/ne4AsfKm+HuIsNjV8eJHQ4RkVXEB3ngodQYKOVS/HKmHnes24PC2jaxw6IBYgJE/dLcqcWy/xn3yFl4dRz83LnwIRE5j/ggDzx2dRxCvFQ4Xd2KW1b/iM8PnhU7LBoAJkDUL3/79gRqWtSIC3BH2nTW/hCR8wn1dsXWx6diYqwf2jR6pH96CL/9IBsVTVwryJ4wAaI+yy5uwIdde+i8OjcJKoVM5IiIiMQR4q3Cxwsn4w83joBMKsE3Rytx/Zu78dbO02jjekF2gQkQ9UljuwaLN+dAEIA7kyMwZViA2CEREYlKJpXg99cPx1e/uwop0b5o1+jxxnf5mPa3nVi3uwBNHSyStmVMgOiyDAYBizfnorS+A5F+rnjx5ivEDomIyGZcEeaFT3+Tir/PG4tofzfUt2nwl29OIHX5Diz94ghyShq4oaoNkosdANm+VdtPYld+DZRyKdY9kMxFD4mILiCVSjB3fARuHROGL3LK8K8fC5Ff1YIP95bgw70liPB1xa1jw3DLmFBcEeoFiYRbB4mNCRBd0r9+PIPV358GALw+dzRGhXmLHBERke2Sy6S4KyUSdyZHIOtMHT7ZV4rteVU429CBtbsKsHZXAaL83HDV8AAIAjAswB1uyov/KL5vUpQVo3cuTIDoojb8VIhXv84DADx5/XDckRwhckRERLbho64JIZdy36QoTBkWgA6NHt+fqMZXh8rxfX41SurbzedLAIR6qxDp54ZwH1eE+7oiyFMFGTeXHnJMgKgHrd6A177Ow8Y9RQCA318Xj8U3DBc3KCIiO+XqIsPNY0Jx85hQtKp1+KWgDj8X1OLrwxWoblGjvKkT5U2d5vZyqQTBXioEeLiguqUTsQHuiA1wR6SvG3zcFBw+sxCJwMqsHpqbm+Ht7Y2mpiZ4eXmJHY5Vlda3I/3TXOwvagAApN84Ar+/Lr7P/+D68lsREREZNXdqUVjbhrKGDpQ1dqC8sQPqS+w47+4iQ7ivq7m3KNzHDRG+xj9H+LgiwEMJqRP3HvXn5zcToF44YwKk1unxrx8L8Y/vT6FTa4CnUo6/zxuHG64INrdhckNENLQMgoD6Vg2qWjpR26qBj6sChbVtKKxrQ02L+rLnu8ikCPNRIcL33JBaSX07fNwU8HVzgZdK0WN4zZHqjPrz81v0IbA1a9bgjTfeQEVFBUaNGoVVq1Zh2rRpF22/e/dupKen49ixYwgLC8PTTz+NtLS0bm22bNmCF198EQUFBRg2bBhee+01zJ07d6hvxS41d2rx8d4SbPi5EFXNxn9ck2L98Jc7xiA2wF3k6IiInItUIkGApxIBnsZNps9PTjq1epQ3GnuKyho6cLahAz+drkVjuwaN7Vo0dWih0RtQVNeOorr2i1wf8HN3QZCnCsFeKgR7KTEhxhcxAe5QyJxrZRxRE6DNmzdj8eLFWLNmDaZOnYq3334bM2fOxPHjxxEV1TMjLSwsxKxZs7Bw4UJ88MEH+Pnnn7Fo0SIEBgbijjvuAABkZWVh3rx5eOWVVzB37lx88cUXuPvuu/HTTz9h0qRJ1r5Fm1TXqsaegjpkHKnA9yeqzd2twV5KPDczEbPHhXGMmYjIBlyq5z3MxxV3p0Sa/643CGju0KKhw5gQmRKjxnYtGto1aOzQQm8QUNuqQW2rBscrmgEAn+wvhUImQVyAB0aEeGJEkPG/I4M9Eenn5rAF2aIOgU2aNAlXXnkl1q5daz6WmJiIOXPmYPny5T3aP/PMM/jyyy+Rl5dnPpaWloZDhw4hKysLADBv3jw0Nzfjm2++Mbf51a9+BV9fX3z88cd9issRhsAEQUC7Ro+Kpk6U1rejoKYVeRUtOFrWhPyqlm5thwd5YOHVcejU6CF3st8AiIichUEQ0NqpQ3WLGlXNnahq7kR1ixr1bRq0XmT7DqVciuHBHhgR5GlMjoI9EObjikAPJXzdXGyu3sguhsA0Gg2ys7Px7LPPdjs+Y8YM7Nmzp9dzsrKyMGPGjG7HbrrpJqxfvx5arRYKhQJZWVlYsmRJjzarVq2yaPwDUdXciS9yyiAIxi9EwLjKsgDj3wXBmLic/3fD+ccuaKvVG9CpNaBTp0enRo8OrR4N7Vo0tGlQ366B5hKFdCODPXFNQiBuGxtmXpSLNT5ERI5LKpHAy1UBL1cF4oM8zMfvnRiJ8qZOnKxswcmqFuRXGf97qqoVap0BR8uacbSsucf1ZFIJ/N1d4OfuAg+lHG5KOTyUMri5yKGQSSGXSiCTSsz/PfdnKeQyCUK8VKIuryJaAlRbWwu9Xo/g4OBux4ODg1FZWdnrOZWVlb221+l0qK2tRWho6EXbXOyaAKBWq6FWnysua2pqAmDMJC0pv7QBr289aNFrXo67UoYIXzdIJECwpxLBXsb1JtyVcgAGZJ04i6wTVg2JiIhsSEtLCzylQHKYCslhKgCBAIxDamcb2nGquhWnq1pRUGN81baoUd+uhQFAZUcbKmsH9rljIrxx43DLjrKYfm73ZXBL9CLoC2tNBEG4ZP1Jb+0vPN7fay5fvhwvv/xyj+ORkZG9tLY/zG+IiOhiFor0uaUAvJ8ammu3tLTA2/vSOxeIlgAFBARAJpP16Jmprq7u0YNjEhIS0mt7uVwOf3//S7a52DUB4LnnnkN6err57waDAfX19fD393f4YuDm5mZERkaitLTUbuudBovPwIjPgc8A4DMA+AxM7PE5CIKAlpYWhIWFXbataAmQi4sLkpOTkZmZ2W2KemZmJmbPnt3rOampqfjqq6+6Hdu2bRtSUlKgUCjMbTIzM7vVAW3btg1Tpky5aCxKpRJKpbLbMR8fn/7ekl3z8vKymy/wocJnYMTnwGcA8BkAfAYm9vYcLtfzYyLqEFh6ejoefPBBpKSkIDU1Fe+88w5KSkrM6/o899xzKCsrw6ZNmwAYZ3z985//RHp6OhYuXIisrCysX7++2+yuJ598EldffTX++te/Yvbs2fjvf/+L7du346effhLlHomIiMj2iJoAzZs3D3V1dVi2bBkqKiqQlJSEjIwMREdHAwAqKipQUnJuZlJsbCwyMjKwZMkSvPXWWwgLC8Pq1avNawABwJQpU/DJJ5/ghRdewIsvvohhw4Zh8+bNXAOIiIiIzEQvgl60aBEWLVrU63sbN27scWz69Ok4ePDSM6nuvPNO3HnnnZYIz+EplUq89NJLPYYAnQmfgRGfA58BwGcA8BmYOPpz4F5gRERE5HS47C8RERE5HSZARERE5HSYABEREZHTYQLkhJYvXw6JRILFixebjwmCgD//+c8ICwuDq6srrrnmGhw7dky8IC3sz3/+MyQSSbdXSEiI+X1Hv//zlZWV4YEHHoC/vz/c3Nwwbtw4ZGdnm9939GcRExPT42tBIpHg8ccfB+D49w8AOp0OL7zwAmJjY+Hq6oq4uDgsW7YMBsO5/QOd4Tm0tLRg8eLFiI6OhqurK6ZMmYL9+/eb33fEZ/DDDz/g1ltvRVhYGCQSCbZu3drt/b7cs1qtxu9//3sEBATA3d0dt912G86ePWvFu7AQgZzKvn37hJiYGGHMmDHCk08+aT7+l7/8RfD09BS2bNkiHDlyRJg3b54QGhoqNDc3ixesBb300kvCqFGjhIqKCvOrurra/L6j379JfX29EB0dLTzyyCPC3r17hcLCQmH79u3C6dOnzW0c/VlUV1d3+zrIzMwUAAg7d+4UBMHx718QBOHVV18V/P39hf/9739CYWGh8J///Efw8PAQVq1aZW7jDM/h7rvvFq644gph9+7dwqlTp4SXXnpJ8PLyEs6ePSsIgmM+g4yMDGHp0qXCli1bBADCF1980e39vtxzWlqaEB4eLmRmZgoHDx4Urr32WmHs2LGCTqez8t0MDhMgJ9LS0iIMHz5cyMzMFKZPn25OgAwGgxASEiL85S9/Mbft7OwUvL29hXXr1okUrWW99NJLwtixY3t9zxnu3+SZZ54Rrrrqqou+70zPwuTJJ58Uhg0bJhgMBqe5/5tvvlmYP39+t2O333678MADDwiC4BxfB+3t7YJMJhP+97//dTs+duxYYenSpU7xDC5MgPpyz42NjYJCoRA++eQTc5uysjJBKpUK3377rdVitwQOgTmRxx9/HDfffDNuuOGGbscLCwtRWVmJGTNmmI8plUpMnz4de/bssXaYQ+bUqVMICwtDbGws7rnnHpw5cwaA89w/AHz55ZdISUnBXXfdhaCgIIwfPx7vvvuu+X1nehYAoNFo8MEHH2D+/PmQSCROc/9XXXUVduzYgZMnTwIADh06hJ9++gmzZs0C4BxfBzqdDnq9HiqVqttxV1dX/PTTT07xDC7Ul3vOzs6GVqvt1iYsLAxJSUl291yYADmJTz75BAcPHsTy5ct7vGfaPPbCDWODg4N7bCxrryZNmoRNmzbhu+++w7vvvovKykpMmTIFdXV1TnH/JmfOnMHatWsxfPhwfPfdd0hLS8MTTzxh3m7GmZ4FAGzduhWNjY145JFHADjP/T/zzDO49957kZCQAIVCgfHjx2Px4sW49957ATjHc/D09ERqaipeeeUVlJeXQ6/X44MPPsDevXtRUVHhFM/gQn2558rKSri4uMDX1/eibeyF6CtB09ArLS3Fk08+iW3btvX4bed8Eomk298FQehxzF7NnDnT/OfRo0cjNTUVw4YNw7///W9MnjwZgGPfv4nBYEBKSgpef/11AMD48eNx7NgxrF27Fg899JC5nTM8CwBYv349Zs6c2WPnaEe//82bN+ODDz7ARx99hFGjRiE3NxeLFy9GWFgYHn74YXM7R38O77//PubPn4/w8HDIZDJceeWVuO+++7rtNuDoz6A3A7lne3wu7AFyAtnZ2aiurkZycjLkcjnkcjl2796N1atXQy6Xm7P9C7P36urqHr8JOAp3d3eMHj0ap06dMs8Gc4b7Dw0NxRVXXNHtWGJionnPPWd6FsXFxdi+fTsWLFhgPuYs9//HP/4Rzz77LO655x6MHj0aDz74IJYsWWLuIXaW5zBs2DDs3r0bra2tKC0txb59+6DVahEbG+s0z+B8fbnnkJAQaDQaNDQ0XLSNvWAC5ASuv/56HDlyBLm5ueZXSkoK7r//fuTm5iIuLg4hISHIzMw0n6PRaLB7925MmTJFxMiHjlqtRl5eHkJDQ83f7Jzh/qdOnYr8/Pxux06ePGnegNiZnsV7772HoKAg3HzzzeZjznL/7e3tkEq7f/uXyWTmafDO8hxM3N3dERoaioaGBnz33XeYPXu20z0DoG//35OTk6FQKLq1qaiowNGjR+3vuYhXf01iOn8WmCAYpz56e3sLn3/+uXDkyBHh3nvvtfvpnuf7wx/+IOzatUs4c+aM8Msvvwi33HKL4OnpKRQVFQmC4Pj3b7Jv3z5BLpcLr732mnDq1Cnhww8/FNzc3IQPPvjA3MYZnoVerxeioqKEZ555psd7znD/Dz/8sBAeHm6eBv/5558LAQEBwtNPP21u4wzP4dtvvxW++eYb4cyZM8K2bduEsWPHChMnThQ0Go0gCI75DFpaWoScnBwhJydHACCsXLlSyMnJEYqLiwVB6Ns9p6WlCREREcL27duFgwcPCtdddx2nwZP9uDABMhgMwksvvSSEhIQISqVSuPrqq4UjR46IF6CFmdayUCgUQlhYmHD77bcLx44dM7/v6Pd/vq+++kpISkoSlEqlkJCQILzzzjvd3neGZ/Hdd98JAIT8/Pwe7znD/Tc3NwtPPvmkEBUVJahUKiEuLk5YunSpoFarzW2c4Tls3rxZiIuLE1xcXISQkBDh8ccfFxobG83vO+Iz2LlzpwCgx+vhhx8WBKFv99zR0SH87ne/E/z8/ARXV1fhlltuEUpKSkS4m8HhbvBERETkdFgDRERERE6HCRARERE5HSZARERE5HSYABEREZHTYQJERERETocJEBERETkdJkBERETkdJgAERERkdNhAkREdq+oqAgSiQS5ubl9PufPf/4zxo0bN2QxEZFt40rQRCQKiUSCL774AnPmzBn0tfR6PWpqahAQEAC5XN6nc1pbW6FWq+Hv7z/ozyci+8MeICKyWVqttk/tZDIZQkJC+pz8AICHh4dDJz8ajUbsEIhsGhMgIif17bff4qqrroKPjw/8/f1xyy23oKCgAACQmpqKZ599tlv7mpoaKBQK7Ny5EwBQUVGBm2++Ga6uroiNjcVHH32EmJgYrFq16rKfHRMTAwCYO3cuJBKJ+e+mYakNGzYgLi4OSqUSgiBcMlag5xDYrl27IJFIsGPHDqSkpMDNzQ1TpkxBfn6++ZwLh8AeeeQRzJkzBytWrEBoaCj8/f3x+OOPd0vCBnPPps+MioqCUqlEWFgYnnjiCfN7arUaTz/9NCIjI6FUKjF8+HCsX7/e/P7u3bsxceJEKJVKhIaG4tlnn4VOpzO/f8011+B3v/sd0tPTERAQgBtvvBEAcPz4ccyaNQseHh4IDg7Ggw8+iNra2j7FS+TImAAROam2tjakp6dj//792LFjB6RSKebOnQuDwYD7778fH3/8Mc4fId+8eTOCg4Mxffp0AMBDDz2E8vJy7Nq1C1u2bME777yD6urqPn32/v37AQDvvfceKioqzH8HgNOnT+PTTz/Fli1bzAnNpWK9lKVLl+LNN9/EgQMHIJfLMX/+/Eu237lzJwoKCrBz5078+9//xsaNG7Fx40bz+4O5588++wx///vf8fbbb+PUqVPYunUrRo8e3e3an3zyCVavXo28vDysW7cOHh4eAICysjLMmjULEyZMwKFDh7B27VqsX78er776arfP+Pe//w25XI6ff/4Zb7/9NioqKjB9+nSMGzcOBw4cwLfffouqqircfffdfYqZyKGJuRU9EdmO6upqAYBw5MgRobq6WpDL5cIPP/xgfj81NVX44x//KAiCIOTl5QkAhP3795vfP3XqlABA+Pvf/96nzwMgfPHFF92OvfTSS4JCoRCqq6v7HKsgCEJhYaEAQMjJyREEQRB27twpABC2b99uPufrr78WAAgdHR3mzxo7dqz5/YcffliIjo4WdDqd+dhdd90lzJs3zyL3/OabbwojRowQNBpNj/fy8/MFAEJmZmav5z7//PPCyJEjBYPBYD721ltvCR4eHoJerxcEQRCmT58ujBs3rtt5L774ojBjxoxux0pLSwUAQn5+/mVjJnJk7AEiclIFBQW47777EBcXBy8vL8TGxgIASkpKEBgYiBtvvBEffvghAKCwsBBZWVm4//77AQD5+fmQy+W48sorzdeLj4+Hr6/voOOKjo5GYGBgn2O9lDFjxpj/HBoaCgCX7LEZNWoUZDJZt3NM7Qd7z3fddRc6OjoQFxeHhQsX4osvvjAPYeXm5kImk5l71y6Ul5eH1NRUSCQS87GpU6eitbUVZ8+eNR9LSUnpdl52djZ27twJDw8P8yshIQEAug0hEjkjJkBETurWW29FXV0d3n33Xezduxd79+4FcK549v7778dnn30GrVaLjz76CKNGjcLYsWMBoNvQ2Pkudrw/3N3d+x3rxSgUCvOfTcnDpYbNzm9vOsfUfrD3HBkZifz8fLz11ltwdXXFokWLcPXVV0Or1cLV1fWS5wqC0C35Of9zzz9+4bMzGAy49dZbkZub2+116tQpXH311X2Km8hRMQEickJ1dXXIy8vDCy+8gOuvvx6JiYloaGjo1mbOnDno7OzEt99+i48++ggPPPCA+b2EhATodDrk5OSYj50+fRqNjY19jkGhUECv11skVmuwxD27urritttuw+rVq7Fr1y5kZWXhyJEjGD16NAwGA3bv3t3reVdccQX27NnTLdnas2cPPD09ER4eftHPu/LKK3Hs2DHExMQgPj6+26u3RJPImTABInJCvr6+8Pf3xzvvvIPTp0/j+++/R3p6erc27u7umD17Nl588UXk5eXhvvvuM7+XkJCAG264AY899hj27duHnJwcPPbYY3B1de3RU3ExMTEx2LFjByorKy+Z0PQlVmsY7D1v3LgR69evx9GjR3HmzBm8//77cHV1RXR0NGJiYvDwww9j/vz52Lp1KwoLC7Fr1y58+umnAIBFixahtLQUv//973HixAn897//xUsvvYT09HRIpRf/Nv7444+jvr4e9957L/bt24czZ85g27ZtmD9/fp+STyJHxgSIyAlJpVJ88sknyM7ORlJSEpYsWYI33nijR7v7778fhw4dwrRp0xAVFdXtvU2bNiE4OBhXX3015s6di4ULF8LT0xMqlapPMbz55pvIzMxEZGQkxo8fP+hYrWEw9+zj44N3330XU6dOxZgxY7Bjxw589dVX5rWI1q5dizvvvBOLFi1CQkICFi5ciLa2NgBAeHg4MjIysG/fPowdOxZpaWl49NFH8cILL1zyM8PCwvDzzz9Dr9fjpptuQlJSEp588kl4e3tfMnEicgZcCZqILOLs2bOIjIzE9u3bcf3114sdjlU44z0TOQomQEQ0IN9//z1aW1sxevRoVFRU4Omnn0ZZWRlOnjzZo5jYUTjjPRM5KvaBEtGAaLVaPP/88xg1ahTmzp2LwMBA7Nq1CwqFAh9++GG3qdfnv0aNGiV26APmjPdM5KjYA0REFtfS0oKqqqpe31MoFIiOjrZyREPPGe+ZyJ4xASIiIiKnwyEwIiIicjpMgIiIiMjpMAEiIiIip8MEiIiIiJwOEyAiIiJyOkyAiIiIyOkwASIiIiKnwwSIiIiInM7/A5xM+XBd3BdYAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Most of them taking single training \n",
    "sns.distplot(train.avg_training_score )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "930cd207-71c9-4709-97c8-b8fdb84c0ae9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 500x300 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 500x300 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAi0AAAEmCAYAAACuxqAsAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy81sbWrAAAACXBIWXMAAA9hAAAPYQGoP6dpAAA2s0lEQVR4nO3deVhU5f8+8HtQGJBlWBQBBUGJlFxQwURQQUtwQdQWFzIItZSQXAIlM1wwlzKtXMoll7Tso5iZGO7gboriiqQIrpimyCIKKM/vD3+cbyOgMIwOx+7Xdc11zTnnOc95P+dYc3OWGYUQQoCIiIiohtPTdQFERERElcHQQkRERLLA0EJERESywNBCREREssDQQkRERLLA0EJERESywNBCREREssDQQkRERLJQW9cF0H9XSUkJrl27BlNTUygUCl2XQ0REOiKEQF5eHuzs7KCnV/H5FIYW0plr167B3t5e12UQEVENcfnyZTRs2LDC5QwtpDOmpqYAHv0jNTMz03E1RESkK7m5ubC3t5c+FyrC0EI6U3pJyMzMjKGFiIieeqsAb8QlIiIiWWBoISIiIlng5SHSuU6f/oxaSiNdl0EvkOQv3tV1CUT0DPBMCxEREckCQwsRERHJAkMLERERyQJDCxEREckCQwsRERHJAkMLERERyQJDCxEREckCQwsRERHJAkMLERERyQJDCxEREckCQwsRERHJAkMLERERyQJDCxEREckCQwsRERHJAkMLERERyQJDCxEREckCQwsRERHJAkPLCygzMxMKhQIpKSm6LoWIiEhrdBpaQkJCoFAoMHz48DLLwsLCoFAoEBISorXtKRQKbNiwQSt9lQaD6khISEDr1q1hZGSEBg0aICwsTCu1ERERvYh0fqbF3t4ea9aswb1796R59+/fx88//wwHBwcdVlax4uLiavdx//599OvXDy1btsTJkycRHx8PNze36hdHRET0gtJ5aGnTpg0cHBywfv16ad769ethb2+P1q1bq7VNSEiAt7c3zM3NYWVlhV69eiE9PV1aXlRUhPDwcNja2sLQ0BCOjo6YPn06AMDR0REA0LdvXygUCmkaAH7//Xe0bdsWhoaGaNy4MSZPnowHDx5IyxUKBb777jsEBgbC2NgYsbGxZcZx8eJFBAQEwMLCAsbGxnjllVewefPmJ469Vq1aCAoKgrOzM9zc3PD+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b2/s3r1be9URERER/X8ahZbGjRsjMzMTAODq6or//e9/AB6dgTE3N9dWbUREREQSjULLe++9h+PHjwMAoqOjpXtbRo8ejcjISK0WSERERARo+D0tj7t06RKOHDmCJk2aoFWrVtqoi/4D+D0tREQEPOPvaXmcg4MDHBwctNEVERERUbk0ujwUERGBb775psz8efPmYdSoUdWtiYiIiKgMjUJLXFwcvLy8yszv0KED1q1bV+2iiIiIiB6n0eWhW7duqf1YYikzMzP8888/1S6K/lte/+511DbSypXKMvaN3PdM+iUioudPozMtzs7OSEhIKDP/jz/+QOPGjatdFBEREdHjNPrzdsyYMQgPD8fNmzfRpUsXAMCOHTswe/ZszJ07V5v1EREREQHQMLSEhoaisLAQ06ZNw9SpUwEAjo6OWLhwId59912tFkhEREQEVOOR5xEjRmDEiBG4efMmjIyMYGJios26iIiIiNRU++7HevXqaaMOIiIioifSKLQ4OTlBoVBUuPzChQsaF0RERERUHo1Cy+NfIFdcXIxjx44hISGBvz1EREREz4RGoeWjjz4qd/78+fNx5MiRahVEREREVB6NvqelIt27d0dcXJw2uyQiIiICoOXQsm7dOlhaWmqzSyIiIiIAGl4eat26tdqNuEIIXL9+HTdv3sSCBQu0VhwRERFRKY1CS58+fdSm9fT0UK9ePfj4+KBp06baqIuIiIhIjUahJSYmRtt1EBERET1RpUNLbm5upTs1MzPTqBgiIiKiilQ6tJibmz/xC+X+7eHDhxoXRERERFSeSoeWXbt2Se8zMzMxfvx4hISEwNPTEwBw4MABrFixAtOnT9d+lURERPSfV+nQ0rlzZ+n9lClT8NVXX2HgwIHSvN69e6NFixZYtGgRgoODtVslERER/edp9D0tBw4cgLu7e5n57u7u+PPPP6tdFD0yadIkuLm5PfPt+Pj4lPlphqpavnw5zM3NtVIPERFReTQKLfb29vjuu+/KzP/+++9hb29f7aKIiIiIHqfRI89z5szBG2+8gS1btqB9+/YAgIMHD+L8+fNYv369VgskIiIiAjQ809KjRw+cO3cOgYGBuH37Nm7duoXAwECcO3cOPXr00HaNOuXo6Ii5c+eqzXNzc8OkSZMAAAqFAkuWLEHfvn1Rp04dvPTSS9i4caPUNjs7G0FBQahXrx6M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3sbs2bPh7u6OY8eOISwsDCNGjMDZs2cBAH/++ScAYPv27cjKysL69eu1Oj6imqa2rgsgIpKjvLw8fP3115g3bx6Cg4MBAE2aNIG3tzcWL16Me/fuYeXKlTA2NgYAzJs3DwEBAZg5cybq169f6e306NEDYWFhAIBx48Zhzpw5SExMRNOmTVGvXj0AgJWVFWxsbLQ8QqKah2daiIg0kJqaisLCQnTt2rXcZa1atZICCwB4eXmhpKQEaWlpVdpOy5YtpfcKhQI2Nja4ceOG5oUTyRhDCxGRBoyMjCpcJoSAQqEod1npfD09PQgh1JYVFxeXaa+vr19m/ZKSkqqWS/RCYGghItLASy+9BCMjI+zYsaPMMldXV6SkpODu3bvSvH379kFPTw8uLi4AgHr16iErK0ta/vDhQ5w6dapKNRgYGEjrEv0XMLQQEWnA0NAQ48aNQ1RUFFauXIn09HQcPHgQS5cuRVBQEAwNDREcHIxTp05h165dGDlyJAYPHizdz9KlSxfEx8cjPj4eZ8+eRVhYGO7cuVOlGqytrWFkZISEhAT8/fffyMnJeQYjJao5GFqIiDQ0ceJEjB07Fp999hmaNWuG/v3748aNG6hTpw62bNmC27dvw8PDA2+++Sa6du2KefPmSeuGhoYiODgY7777Ljp37gwnJyf4+vpWafu1a9fGN998g++//x52dnYIDAzU9hCJahSFePyiKhEREVENxDMtREREJAsMLURERCQLDC1EREQkCwwtREREJAsMLURERCQLDC1EREQkCwwtREREJAsMLURERCQLDC1EREQkCwwtREREJAsMLURERCQLDC1EREQkC/8PEoZUKEWSuEwAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 500x300 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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7TXFx8crIiJCvXv31oEDB2rcR2VlpSZOnKiYmBhFRkbq6aef1meffVZjmwsXLmj06NFyuVxyuVwaPXq0Ll68eMd5P/roIw0ZMkTx8fFyOBz65z//WWN9feY9deqUhgwZosjISMXExOgXv/iFrl69elf5x4wZU+v16N69e0DkX7hwoR599FFFRUWpZcuWGjZsmAoKCmpsE8jjfzv5A3n8//CHPygtLc17IYAePXrogw8+8K4P5LG/nfyBPPZ1WbhwoRwOhyZPnuxdFuivQS2mgVi7dq0JDw83K1euNAcPHjSTJk0ykZGR5uTJk/WaY968eebb3/62OXPmjPdWUlLiXb9o0SITFRVl1q1bZ/Lz882IESNM69atTVlZmXebl156ySQkJJjNmzcbt9tt+vTpY7p27WquX7/u3WbgwIEmJSXF7Ny50+zcudOkpKSYwYMH33He999/3/zyl78069atM5LMhg0baqyvr7zXr183KSkppk+fPsbtdpvNmzeb+Ph4k5mZeVf5MzIyzMCBA2u8HufPn6+xjb/yDxgwwKxatcrs37/f7N271wwaNMi0bdvWXLp0KSjG/3byB/L4v/fee2bjxo2moKDAFBQUmFdffdWEh4eb/fv3B/zY307+QB77G+3atcs88MADJi0tzUyaNMm7PNBfgxs1mEJ97LHHzEsvvVRjWZcuXczMmTPrNce8efNM165d61zn8XhMXFycWbRokXfZlStXjMvlMsuXLzfGGHPx4kUTHh5u1q5d692msLDQhISEmE2bNhljjDl48KCRZHJzc73b5OTkGEnm8OHDPme/sZDqM+/7779vQkJCTGFhoXebv/3tb8bpdJrS0lKf8hvz1R+VoUOH3nSfQMpfUlJiJJnt27cbY4Jv/G/Mb0xwjb8xxjRr1sy88847QTf2N+Y3JnjGvry83HTs2NFs3rzZ9OrVy1uowfgaNIi3fK9evaq8vDz179+/xvL+/ftr586d9Z7nyJEjio+PV/v27fXss8/q2LFjkqTjx4+ruLi4Rk6n06levXp5c+bl5enatWs1tomPj1dKSop3m5ycHLlcLj3++OPebbp37y6Xy2X1+dZn3pycHKWkpNS4kPWAAQNUWVmpvLy8u3oeWVlZatmypTp16qQXXnhBJSUl3nWBlL+0tFTS/77gIdjG/8b81YJh/KuqqrR27VpdvnxZPXr0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",
      "text/plain": [
       "<Figure size 500x300 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 500x300 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "for col in train.select_dtypes(include='object').columns:\n",
    "    plt.figure(figsize=(5,3))\n",
    "    sns.countplot(y=train[col])\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4f3dc508-8bbb-42df-a54a-3f4cbe9ac52f",
   "metadata": {},
   "outputs": [],
   "source": [
    "#HR and Leagal Dept having less strength\n",
    "#43.08% of employees belongs to Region 2,22 and7\n",
    "#More employees recruitment thru other channel\n",
    "# 29.76 % are female and above 70% are male\n",
    "# UG categories are high\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "81587a0d-361f-496f-88b1-2b0cd1a5dd88",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: ylabel='is_promoted,avg_training_score'>"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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OyGEEOIK4e3i6bLHdqc2BnYbOHAgH374IZGRkSoy03VdLZy8OfMUFhaG2WxmyJAhLF++nClTpgAQEFC5mOnZsyc+Pj74+Phw9913895776kvatSoUaqvnrsxMCCxnRBCCCH+GrFdQEAAP/74I126dAHOH9tNnz6d7du389FHH1U5d3h4OAaDgfXr12O323nkkUf47bffCAsLU6UITCYTnTp1Yv369UBlcc6kpCSaNGnC3r17AYiOjiY/P5/i4mKJ7YQQQoha7IookqlpGgaDgbi4OLVwgspFzKmxnbvXnDu2GzFiBJqm0blz53M2Bs7JyaFly5ZA5S5XTEwM6enpOBwOTCYTDRo0oFWrVgC8+uqrqqp5UVGRmmfixIkqtqvubTshhBBC1D6XdfF0emznjscGDhxITEwMEydOJDc3t0alCho3bsyxY8dYsmQJTqeTgQMHsmfPHurWrasOnXft2pUtW7YQEBBASUkJBoOBpk2bUq9ePTWPO7arqKiosjFwVb3tzleqAOTckxBCCHGluayLJ3evuOHDh3s0Bh49ejRFRUXq3FFNGgPv27cPHx8fVqxYQU5ODk2aNOHo0aM0bNhQjXnyySfp0qULQUFBPPvss3zwwQfs27ePRx55RI2pX78+FosFm81WZWPgcePGqbpQ8GdsJ42BhRBCiNrnst+227dvH2az+aI3Bg4PD+eRRx7hp59+4vfffyc8PJyoqChsNhsHDx5U43r27MnWrVvVYs1sNnP33XfzySefAJUFO7dv364OrXvD28xUCCGEEH8dV8SZJ3ez3UtRYdzHx4fJkyer3/Py8sjLy/OI3SIiIjh+/LjH64qLiz0WblarldTUVFwuV5Wx3dkOjIN3sR1IdCeEEEJcSS7r4sliseB0OlVjYHds524MfODAAQoKCmoU2/3666+MHj2aFStWoGkaTZo04eDBg2rBlpmZid1u591336Vr164UFRUxePBgcnJyGDZsmJonPj6exYsXA1QZ21V15kliOyGEEKL2uayxXbdu3UhKSiIwMPCix3YAu3fvJiYmhrlz53L06FHGjh2retV5WySzffv2bNu2DYPBoM5onU5KFQghhBBXvisittN1HYPBcEliO4DIyEhGjhzJU089xdGjRzEYDLRr1w6oPOv06KOP8vHHH3u8xsfHhz/++EMdVrdarezcuVPtWFWHxHZC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iBotntwlBgDq1aunGvYGBwerv3vD3Rg4JiaGpk2bApX1o0wmE7t376Zjx47ccsstREREAH/Gdu7XnqtUgcvlYvTo0djtdt5//31OnjyJw+HwuEnXoEEDYmJiGD9+PIMGDSIvLw+ADRs2qDEjR47EbrcD1W8M3OalFTQd+321XiOEEEKIv7YaLZ46dOig2pv06NGDF198kS+//JInnniCtm3bev/mVTQGdjgctGjRgi1btrBy5UpycnKA6lUYv/XWW3nnnXdITk7m1VdfpXXr1pSXl3vscNntdgYMGMCCBQuAyj52wcHBqhgnSGNgIYQQQniq0eLptddeo0GDBgBMmjSJ8PBwHn30UXJycvj000+9nse9kMnJySE5Odnzg11gY+AHHniAgIAAevToQXZ2Nrt27cLHx0dFdgC//vort956Kx07dgRg586dFBYWct9996kxCxYswM/PD6h+bLdjwq0ceL2v19+HEEIIIf76anTbrlOnTurvERERLFu2rEZv7o7t4uLiVNuU02M7TdNo2bIl+fn5Ho2BNU2jc+fOZy1VAJXRYmpqKvBnMc6wsDDCw8PVmK1bt/Lee+9RUlICQGhoKLNmzeKWW25RYyZOnKjKI8htOyGEEELUuLfdRXnzSxjbff7555SWljJ37lz27t1Lhw4dOHbsGNddd50aM2rUKFq1akVYWBgA999/v8fCCS4stnOfeZJzT0IIIUTtcdEXTzfddBOTJk2iuLj4vGMvVWxXWFjIvHnzuOWWW7juuuvIysri0KFDGI1GioqK1LilS5eyZ88evv32WwCKioo4evSo2omCixPbSXQnhBBC1CL6Rfavf/1L7969u964cePzjo2JidENBoPepUsX9dg//vEP3WQy6ZGRkTqga5qmx8TE6ID+9ttvq3GAPmvWrLPO+8cff+hAlT979uxRc5zt59R5P//8c/W4t2w2mw7oNpvN69cIIYQQ4vLy9v/vi97tdtasWQAeOzxVOT22MxgMHrHd8uXLycvLY9CgQYD3sV1sbCy33XYb6enpvPDCC4wZM4bS0lIKCwtVkUyovJE3ePBgOnfuTJs2bWjevDlOp5O7775bzeWO7SoqKtRnPN3Z2rNAZWxnMAecMV52ooQQQogr10VfPLkFBQWdd4w7ths+fDhZWVkeV/wvJLbz8/Pjiy++YNiwYQwdOhSDwYDRaFQH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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "train[['avg_training_score','is_promoted']].groupby('is_promoted').value_counts().plot(kind='barh')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "5ba2aff1-bd99-4ae2-9d44-0fb7e4aacfc2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: ylabel='KPIs_met >80%,is_promoted'>"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAlEAAAGdCAYAAAAyviaMAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy81sbWrAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAxDUlEQVR4nO3de3RU1d3/8c+EJAMiiYFAQkIkXCIaA5oCYoKritQEw0UepKBchFpEqhSsoj9pRcRLAS8UtTx9qqLoowUs98olAYUAGsFyURKQixCJkRANIQGBgMn+/eFino4JOucww8yQ92utWYvsc2af72zOWvNZe/bscRhjjAAAAGBJiL8LAAAACEaEKAAAABsIUQAAADYQogAAAGwgRAEAANhAiAIAALCBEAUAAGADIQoAAMCGUH8XEGxqamr09ddfq0mTJnI4HP4uBwAAeMAYo2PHjikuLk4hId6ZQyJEWfT1118rISHB32UAAAAbioqK1KpVK6/0RYiyqEmTJpJ++E+IiIjwczUAAMATlZWVSkhIcL2PewMhyqKzH+FFREQQogAACDLeXIrDwnIAAAAbCFEAAAA2EKIAAABsIEQBAADYQIgCAACwgRAFAABgAyEKAADABkIUAACADYQoAAAAGwhRAAAANhCiAAAAbCBEAQAA2ECIAgAAsIEQBQAAYAMhCgAAwAZCFAAAgA2EKAAAABsIUQAAADYQogAAAGwgRAEAANgQ6u8CglXK5GyFOC/xdxkeK5zW298lAABwUWEmCgAAwAZCFAAAgA2EKAAAABsIUQAAADYQogAAAGwgRAEAANhAiAIAALCBEAUAAGADIQoAAMAGQhQAAIANhCgAAAAbCFEAAAA2EKIAAABsIEQBAADYQIgCAACwgRAFAABgAyEKAADABr+HqLKyMrVo0UKFhYUX/NoDBw7UjBkzLvh1AQBA8PN7iJo6dar69u2rxMREV9v48ePVuXNnOZ1OXXvttbb6LSgo0O23367ExEQ5HA7NnDmz1jmPP/64nnnmGVVWVtorHgAA1Ft+DVEnT57U7NmzNWrUKLd2Y4zuvvtuDR482HbfJ06cUNu2bTVt2jTFxsbWeU6nTp2UmJiod955x/Z1AABA/RTqz4uvXLlSoaGhSktLc2t/6aWXJEnffPONPvvsM1t9d+3aVV27dpUkPfroo+c8r1+/fpo7d65+97vf2boOAACon/w6E7V+/Xp16dLFnyXouuuu0+bNm1VVVVXn8aqqKlVWVro9AAAA/BqiCgsLFRcX588SFB8fr6qqKpWUlNR5fOrUqYqMjHQ9EhISLnCFAAAgEPl9TVTDhg39WYIaNWok6Yc1VHWZOHGiKioqXI+ioqILWR4AAAhQfl0TFR0drfLycn+WoCNHjkiSmjdvXudxp9Mpp9N5IUsCAABBwK8zUampqdq5c6c/S1B+fr5atWql6Ohov9YBAACCi19DVGZmpgoKCmrNRu3bt0/bt29XSUmJTp48qe3bt2v79u06ffq0x32fPn3a7XnFxcXavn279u3b53behg0blJGR4ZXXAwAA6g+HMcb4s4C0tDSNHDlS9957r6vtpptuUm5ubq1zDxw44NqU0+Fw6I033tDIkSPr7LewsFBt2rSp1X7jjTdq3bp1kqRTp04pJiZG2dnZuv766z2qt7Ky8ocF5g+8qxDnJR49JxAUTuvt7xIAAPCbs+/fFRUVioiI8Eqfft+xfNKkSXrxxRdVU1Pjalu3bp2MMbUeZwNUYWGhQkND1b1793P2m5iYWGcfZwOUJM2ePVvdunXzOEABAACc5deF5ZKUlZWlvXv3qri42OPtA1atWqXRo0crKSnpvK4dFhaml19++bz6AAAA9ZPfP84LNnycBwBA8LkoP84DAAAIRoQoAAAAGwhRAAAANhCiAAAAbCBEAQAA2ECIAgAAsIEQBQAAYAMhCgAAwAZCFAAAgA2EKAAAABsIUQAAADYQogAAAGwgRAEAANhAiAIAALAh1N8FBKv8KZmKiIjwdxkAAMBPmIkCAACwgRAFAABgAyEKAADABkIUAACADYQoAAAAGwhRAAAANhCiAAAAbCBEAQAA2ECIAgAAsIEQBQAAYAMhCgAAwAZCFAAAgA2EKAAAABsIUQAAADYQogAAAGwgRAEAANhAiAIAALAh1JOToqKi5HA4POrwyJEj51UQAABAMPAoRM2cOdP177KyMj399NPKzMxUWlqaJCkvL0/Z2dmaNGmST4oEAAAINA5jjLHyhNtvv109evTQ2LFj3dr/+te/as2aNVqyZIk36ws4lZWVioyMVEVFhSIiIvxdDgAA8IAv3r8tr4nKzs5Wr169arVnZmZqzZo1XikKAAAg0FkOUc2aNdPixYtrtS9ZskTNmjXzSlEAAACBzqM1Uf9pypQp+u1vf6t169a51kR9/PHHWrVqlV577TWvFwgAABCILIeokSNH6qqrrtJLL72kRYsWyRij5ORkffjhh+rWrZsvagQAAAg4lheW13csLAcAIPgExMJySfriiy/02GOPaciQISotLZUkrVq1SgUFBV4pCgAAINBZDlG5ubnq2LGjNm3apIULF+r48eOSpM8++0yTJ0/2eoEAAACByHKIevTRR/X0009r9erVCg8Pd7X36NFDeXl5Xi0OAAAgUFkOUTt27NB//dd/1Wpv3ry5ysrKvFIUAABAoLMcoi677DIdOnSoVvu2bdsUHx/vlaIAAAACneUQNWTIEP2///f/VFJSIofDoZqaGn344YeaMGGC7rrrLl/UCAAAEHAsh6hnnnlGl19+ueLj43X8+HElJyfrl7/8pdLT0/XYY4/5okYAAICAY3ufqP3792vr1q2qqalRamqqkpKSvF1bQGKfKAAAgk9A7BP15JNP6sSJE2rbtq0GDhyoQYMGKSkpSSdPntSTTz7plaIAAAACneWZqAYNGujQoUNq0aKFW3tZWZlatGih6upqrxYYaJiJAgAg+ATETJQxRg6Ho1b7p59+qqZNm3qlKAAAgEDn8Q8QR0VFyeFwyOFw6IorrnALUtXV1Tp+/LjGjBnjkyIBAAACjcchaubMmTLG6O6779aUKVMUGRnpOhYeHq7ExESlpaX5pEgAAIBA43GIGjFihCSpTZs2Sk9PV1hYmM+KAgAACHQeh6izbrzxRlVXV2vhwoXatWuXHA6HkpOT1a9fPzVo0MAXNQIAAAQcyyFq3759ysrKUnFxsTp06CBjjPbs2aOEhAQtX75c7dq180WdAAAAAcXyt/PGjRundu3aqaioSFu3btW2bdt08OBBtWnTRuPGjfNFjQAAAAHH8kxUbm6uPv74Y7ftDJo1a6Zp06ape/fuXi0OAAAgUFmeiXI6nTp27Fit9uPHjys8PNwrRQEAAAQ6yyGqT58+Gj16tDZt2iRjjIwx+vjjjzVmzBj169fPFzUCAAAEHMsf57300ksaMWKE0tLSXNscfP/99+rXr59efPFFrxcYqFImZyvEeYm/ywDqjcJpvf1dAgC4sRyiLrvsMi1dulR79+7V559/LmOMkpOT1b59e1/UBwAAEJAsh6izkpKSlJSU5M1aAAAAgoblEGWM0YIFC7R27VqVlpaqpqbG7fiiRYu8VhwAAECgshyixo8fr1deeUU9evRQTEyM2w8RAwAA1BeWQ9Tbb7+tRYsWKSsryxf1AAAABAXLWxxERkaqbdu2vqgFAAAgaFgOUU888YSmTJmikydP+qIeAACAoGD547xf//rXmjt3rlq0aKHExETXXlFnbd261WvFAQAABCrLIWrkyJHasmWLhg0bxsJyAABQb1kOUcuXL1d2drZuuOEGX9QDAAAQFCyviUpISFBERIQvagEAAAgalkPUCy+8oEceeUSFhYU+KAcAACA4WP44b9iwYTpx4oTatWunSy65pNbC8iNHjnitOAAAgEBlOUTNnDnTB2UAAAAEF8shasSIEb6oAwAAIKhYDlGSVF1drSVLlmjXrl1yOBxKTk5Wv3791KBBA2/XBwAAEJAsh6h9+/YpKytLxcXF6tChg4wx2rNnjxISErR8+XK1a9fOF3UCAAAEFMvfzhs3bpzatWunoqIibd26Vdu2bdPBgwfVpk0bjRs3znIBZWVlatGihV++7Tdw4EDNmDHjgl8XAAAEP8shKjc3V88++6yaNm3qamvWrJmmTZum3NxcywVMnTpVffv2VWJioqtt/Pjx6ty5s5xOp6699lrLfZ61cOFCJScny+l0Kjk5WYsXL3Y7/vjjj+uZZ55RZWWl7WsAAID6yXKIcjqdOnbsWK3248ePKzw83FJfJ0+e1OzZszVq1Ci3dmOM7r77bg0ePNhqeS55eXkaPHiwhg8frk8//VTDhw/XoEGDtGnTJtc5nTp1UmJiot555x3b1wEAAPWT5RDVp08fjR49Wps2bZIxRsYYffzxxxozZoz69etnqa+VK1cqNDRUaWlpbu0vvfSS7r//frVt29ZqeS4zZ87ULbfcookTJ+rKK6/UxIkT1bNnz1pbNPTr109z5861fR0AAFA/WQ5RL730ktq1a6e0tDQ1bNhQDRs2VPfu3dW+fXu9+OKLlvpav369unTpYrUEj+Tl5SkjI8OtLTMzUx999JFb23XXXafNmzerqqqqzn6qqqpUWVnp9gAAALD07TxjjCoqKjR37lx9/fXX2rVrl4wxSk5OVvv27S1fvLCwUHFxcZaf54mSkhLFxMS4tcXExKikpMStLT4+XlVVVSopKVHr1q1r9TN16lRNmTLFJzUCAIDgZTlEJSUlqaCgQElJSbaC0386efKkGjZseF59/BSHw+H2tzGmVlujRo0kSSdOnKizj4kTJ+rBBx90/V1ZWamEhAQvVwoAAIKNpRAVEhKipKQklZWVKSkp6bwvHh0drfLy8vPupy6xsbG1Zp1KS0trzU6d/a2/5s2b19mP0+mU0+n0SY0AACB4WV4T9eyzz+rhhx9Wfn7+eV88NTVVO3fuPO9+6pKWlqbVq1e7teXk5Cg9Pd2tLT8/X61atVJ0dLRP6gAAABcnyzuWDxs2TCdOnNA111yj8PBw18dhZ52d2fFEZmamJk6cqPLyckVFRbna9+3bp+PHj6ukpEQnT57U9u3bJUnJyckeb6Mwfvx4/fKXv9T06dN12223aenSpVqzZo02btzodt6GDRtqLUAHAAD4OZZD1I+3CDgfHTt2VJcuXfTuu+/q3nvvdbWPGjXKbePO1NRUSdKBAwdcm3I6HA698cYbGjlyZJ19p6ena968eXrsscc0adIktWvXTvPnz1e3bt1c55w6dUqLFy9Wdna2114TAACoHxzGGOPPAlasWKEJEyYoPz9fISGefbpYWFiopKQk7dy587zWZs2aNUtLly5VTk6Ox8+prKxUZGSkEh54VyHOS2xfG4A1hdN6+7sEAEHs7Pt3RUWFIiIivNKn5ZkoSaqurtbixYu1a9cuORwOXXXVVbrtttsUGmq9u6ysLO3du1fFxcUef+tt1apVGj169Hkvbg8LC9PLL798Xn0AAID6yfJMVH5+vm677TaVlJSoQ4cOkqQ9e/aoefPmWrZsmTp27OiTQgMFM1GAfzATBeB8+GImyvK380aNGqWrr75aX331lbZu3aqtW7eqqKhInTp10ujRo71SFAAAQKCz/Pnbp59+qn//+99u36aLiorSM888o65du3q1OAAAgEBleSaqQ4cOOnz4cK320tLS897BHAAAIFhYDlF//vOfNW7cOC1YsEBfffWVvvrqKy1YsEAPPPCApk+fzg/1AgCAesHyx3l9+vSRJA0aNMj1O3Rn16b37dvX9bfD4VB1dbW36gQAAAgolkPU2rVrfVEHAABAULEcom688UaPzrvvvvt09dVX85t0AADgomR5TZSn3n77bdZFAQCAi5bPQpSff00GAADAp3wWogAAAC5mhCgAAAAbCFEAAAA2EKIAAABs8FmIGjZsmNd+JRkAACDQOIzFr9GtWrVKl156qW644QZJ0qxZs/Tqq68qOTlZs2bNcvth4otRZWWlIiMjVVFRQUgEACBI+OL92/JM1MMPP+za/2nHjh166KGHlJWVpf379+vBBx/0SlEAAACBzvKO5QcOHFBycrIkaeHCherTp4/+/Oc/a+vWrcrKyvJ6gQAAAIHI8kxUeHi4Tpw4IUlas2aNMjIyJElNmzZlh3IAAFBvWJ6JuuGGG/Tggw+qe/fu2rx5s+bPny9J2rNnj1q1auX1AgEAAAKR5Zmov/71rwoNDdWCBQv0t7/9TfHx8ZKklStXqlevXl4vEAAAIBBZ/nZefce38wAACD6+eP/26OO8yspK1wV/bt0TwQIAANQHHoWoqKgoHTp0SC1atNBll10mh8NR6xxjjBwOh6qrq71eJAAAQKDxKER98MEHatq0qSRp7dq1Pi0IAAAgGLAmyiLWRAEAEHwCYsfyn3Lw4EE+zgMAAPWCV0NUYmKikpOTtWjRIm92CwAAEHAsb7b5U9auXasDBw5owYIFGjBggDe7BgAACCisibKINVEAAAQfv+0TdS7ffvutNm3apOrqanXt2lUtW7b0SlEAAACBznaIWrhwoX7729/qiiuu0JkzZ7R7927NmjVLv/nNb7xZHwAAQEDyeGH58ePH3f6eMmWKNm/erM2bN2vbtm365z//qT/96U9eLxAAACAQeRyiOnfurKVLl7r+Dg0NVWlpqevvw4cPKzw83LvVAQAABCiPF5YXFhbqvvvuk9Pp1KxZs/TFF1/ojjvuUHV1tb7//nuFhIRozpw5ysrK8nXNfsXCcgAAgo9fF5YnJiZqxYoV+sc//qEbb7xR48eP1759+7Rv3z5VV1fryiuvVMOGDb1SFAAAQKCzvNnmkCFDXOugbrrpJtXU1Ojaa68lQAEAgHrF0rfzVq5cqZ07d+qaa67R7NmztW7dOg0ZMkRZWVl68skn1ahRI1/VCQAAEFA8nol65JFHNHLkSH3yySe699579dRTT+mmm27Stm3b5HQ6de2112rlypW+rBUAACBgeLywPDo6WtnZ2ercubOOHDmi66+/Xnv27HEdLygo0L333quNGzf6rNhAwMJyAACCjy/evz2eibrkkkt04MABSVJRUVGtNVBXX331RR+gAAAAzvI4RE2dOlV33XWX4uLidOONN+qpp57yZV0AAAABzdIPEJeVlWn//v1KSkrSZZdd5sOyAhcf5wEAEHz8/gPEzZo1U7NmzbxyYQAAgGBmaZ+oTz75REOHDlWbNm3UqFEjXXLJJWrTpo2GDh2qf//7376qEQAAIOB4PBO1ZMkSDRo0SD179tT48eMVExMjY4xKS0uVk5Oj7t27691339Vtt93my3oBAAACgsdrolJSUjRs2DA9+uijdR6fPn263nrrLRUUFHi1wEDDmigAAIKPX7c42LdvnwYMGHDO4/3799cXX3zhlaIAAAACncchql27dlqyZMk5jy9dulRt27b1Rk0AAAABz+M1UU8++aTuuOMO5ebmKiMjQzExMXI4HCopKdHq1auVk5OjefPm+bJWAACAgOFxiLr99tu1fv16vfjii5oxY4ZKSkokSbGxsUpLS1Nubq7S0tJ8VigAAEAgsbRPVFpaGkEJAABAFveJqsvhw4dds1IAAAD1hcch6siRI7r99tvVunVr3X///aqurtaoUaPUsmVLxcfHKz09XYcOHfJlrQAAAAHD4xA1YcIE7dmzRw8//LAKCgo0cOBAffLJJ9qwYYM2btyo77///px7SAEAAFxsPN5sMy4uTgsWLFB6eroOHz6sli1bKjs7W7fccosk6cMPP9TgwYP11Vdf+bRgf2OzTQAAgo9fN9usqKhQfHy8JCkmJkahoaFq2bKl63hcXJyOHj3qlaIAAAACncchKikpSe+9954kaeXKlWrYsKFycnJcx7Ozs9WmTRvvVwgAABCAPN7i4OGHH9aIESM0c+ZMffXVV3r77bc1btw4bdq0SSEhIVq0aJFmzJjhy1oBAAAChschaujQoWrdurU2bdqk9PR0paWl6aqrrtK0adN04sQJvfLKKxoxYoQvawUAAAgYHi8sxw9YWA4AQPDx68JyAAAA/B9LP/vynwoKCjR8+HC98sor6tKlizdrCgopk7MV4rzkvPspnNbbC9UAAIALzfZM1Jw5c/Tpp5/q9ddf92Y9AAAAQcFWiKqurtY//vEP/eEPf9D8+fN1+vRpb9cFAAAQ0GyFqOzsbH3//feaOnWqIiIitHTpUm/XBQAAENBshag333xTd9xxh8LCwjR06FDNmTPHy2UBAAAENssh6ujRo/rXv/6lu+66S5I0fPhwrV69WocPH/Z6cQAAAIHKcoiaN2+e2rRpo86dO0uSOnTooF/84hd6++23vV4cAABAoLIcot58800NHz7crW3YsGF8pAcAAOoVSyGqqKhI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V7/6lSTJGMOPDwMAgHrD8kzUgAEDNGTIECUlJamsrEy33nqrJGn79u1q37691wsEAAAIRJZD1F/+8hclJiaqqKhIzz77rC699FJJP3zMd99993m9QAAAgEDkMMYYfxcRTCorKxUZGamKigp2LAcAIEj44v3b45moZcuWeXRev379bBcDAAAQLDwOUf379//ZcxwOB4vLAQBAveBxiKqpqfFlHQAAAEHF8hYHVVVV+u6773xRCwAAQNDwOER9++236t27ty699FJFREQoPT1d+/fv92VtAAAAAcvjEDVx4kRt2bJFU6ZM0XPPPadvv/1W9957ry9rAwAACFger4nKzs7W66+/rqysLElSVlaWUlJSdObMGYWFhfmsQAAAgEDk8UzU119/rdTUVNffV155pcLDw/X111/7pDAAAIBA5nGIMsYoNNR94io0NJRv7QEAgHrJ44/zjDHq2bOnW5A6ceKE+vbtq/DwcFfb1q1bvVshAABAAPI4RE2ePLlW22233ebVYgAAAILFeYWoH6usrDyvYgAAAIKFx2uinn/++Z88XllZqYyMjPMuCAAAIBh4HKImTZqkN954o85jx44dU2ZmJjNRAACg3vA4RP3v//6v7rvvPi1ZssSt/fjx48rMzNSRI0e0du1ab9cHAAAQkDxeEzVw4EAdPXpUQ4YM0fLly9WjRw8dP35cvXr10rfffqvc3FzFxMT4slYAAICA4XGIkqRRo0bpyJEj6t+/v5YuXapJkyappKREubm5atmypa9qBAAACDiWQpQkPfLIIyovL1fPnj2VmJio3NxcxcfH+6I2AACAgOVxiBowYIDb32FhYYqOjta4cePc2hctWuSdygAAAAKYxyEqMjLS7e8777zT68UAAAAEC49D1Lm2NwAAAKiPPN7iAAAAAP+HEAUAAGADIQoAAMAGQhQAAIANhCgAAAAbCFEAAAA2EKIAAABsIEQBAADYQIgCAACwgRAFAABgAyEKAADABkIUAACADYQoAAAAGwhRAAAANoT6u4BglTI5WyHOS/xdBgAAF43Cab39XYIlzEQBAADYQIgCAACwgRAFAABgAyEKAADABkIUAACADYQoAAAAGwhRAAAANhCiAAAAbCBEAQAA2ECIAgAAsIEQBQAAYAMhCgAAwAZCFAAAgA2EKAAAABsIUQAAADYQogAAAGwgRAEAANjg9xBV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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "train[['KPIs_met >80%','is_promoted']].groupby('KPIs_met >80%').value_counts().plot(kind='barh')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "b212c190-7801-4a08-8944-082587f48d3f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='previous_year_rating,is_promoted'>"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjoAAAHVCAYAAAD4slEKAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy81sbWrAAAACXBIWXMAAA9hAAAPYQGoP6dpAABBeElEQVR4nO3de1xUdeL/8feAgkgwgsQt8fItZVXMvKSilddAE7WszMUl2Qy7KZlaabv7k9pN/aZmu/qtdV3TUoq2Ld3KwkvmLa9hlLdMWy+oIKYwiNFAeH5/uJ51BC/oIM7x9Xw8ziPmnM858z7HMd6eOWfGZhiGIQAAAAvyqukAAAAA1YWiAwAALIuiAwAALIuiAwAALIuiAwAALIuiAwAALIuiAwAALKtWTQeoSadOndLhw4cVEBAgm81W03EAAMAlMAxDJ06cUGRkpLy8LnzO5rouOocPH1ZUVFRNxwAAAJchJydHDRo0uOCY67roBAQESDp9oAIDA2s4DQAAuBRFRUWKiooyf49fyHVddM68XRUYGEjRAQDAw1zKZSdcjAwAACyLogMAACyLogMAACyLogMAACyLogMAACyLogMAACyrykVn9erV6tevnyIjI2Wz2bRo0SKX5TabrdJpypQp5phu3bpVWD548GCX7RQUFCgpKUl2u112u11JSUkqLCx0GXPgwAH169dP/v7+CgkJUWpqqkpLS6u6SwAAwKKqXHROnjyp1q1ba+bMmZUuz83NdZnefPNN2Ww23X///S7jUlJSXMbNmjXLZXliYqKys7OVmZmpzMxMZWdnKykpyVxeXl6uvn376uTJk1q7dq0yMjL0wQcfaMyYMVXdJQAAYFFV/sDAPn36qE+fPuddHh4e7vL4X//6l7p3767/+Z//cZlft27dCmPP2LlzpzIzM7VhwwZ17NhRkjR79mzFxsZq165dio6O1tKlS7Vjxw7l5OQoMjJSkjRt2jQlJyfr5Zdf5gMAAQBA9V6jc+TIES1evFjDhg2rsCw9PV0hISFq2bKlxo4dqxMnTpjL1q9fL7vdbpYcSerUqZPsdrvWrVtnjomJiTFLjiTFx8fL6XQqKyur0jxOp1NFRUUuEwAAsK5q/QqIt956SwEBARo4cKDL/CFDhqhJkyYKDw/Xtm3bNH78eH3zzTdatmyZJCkvL0+hoaEVthcaGqq8vDxzTFhYmMvyoKAg+fj4mGPONWnSJL344ovu2DUAAOABqrXovPnmmxoyZIjq1KnjMj8lJcX8OSYmRk2bNlX79u21ZcsWtW3bVlLl319hGIbL/EsZc7bx48dr9OjR5uMzXwoGAACsqdreulqzZo127dqlRx999KJj27Ztq9q1a2v37t2STl/nc+TIkQrjjh49ap7FCQ8Pr3DmpqCgQGVlZRXO9Jzh6+trfoEnX+QJAID1VVvRmTNnjtq1a6fWrVtfdOz27dtVVlamiIgISVJsbKwcDoc2bdpkjtm4caMcDoc6d+5sjtm2bZtyc3PNMUuXLpWvr6/atWvn5r0BAACeqMpvXRUXF2vPnj3m47179yo7O1vBwcFq2LChpNNvCb3//vuaNm1ahfV/+OEHpaen65577lFISIh27NihMWPGqE2bNurSpYskqXnz5urdu7dSUlLM286HDx+uhIQERUdHS5Li4uLUokULJSUlacqUKTp+/LjGjh2rlJQUztQAAABJl1F0vvrqK3Xv3t18fOaal6FDh2revHmSpIyMDBmGoV//+tcV1vfx8dHnn3+uP//5zyouLlZUVJT69u2rCRMmyNvb2xyXnp6u1NRUxcXFSZL69+/v8tk93t7eWrx4sZ588kl16dJFfn5+SkxM1NSpU6u6SwCuc43HLXb7NvdN7uv2bQKoOpthGEZNh6gpRUVFstvtcjgcnAUCrmMUHcCzVOX3N991BQAALIuiAwAALIuiAwAALIuiAwAALIuiAwAALIuiAwAALIuiAwAALIuiAwAALIuiAwAALIuiAwAALIuiAwAALIuiAwAALIuiAwAALIuiAwAALIuiAwAALIuiAwAALIuiAwAALIuiAwAALIuiAwAALIuiAwAALIuiAwAALIuiAwAALIuiAwAALIuiAwAALIuiAwAALIuiAwAALIuiAwAALIuiAwAALIuiAwAALIuiAwAALIuiAwAALIuiAwAALIuiAwAALIuiAwAALIuiAwAALIuiAwAALIuiAwAALIuiAwAALIuiAwAALIuiAwAALKvKRWf16tXq16+fIiMjZbPZtGjRIpflycnJstlsLlOnTp1cxjidTo0cOVIhISHy9/dX//79dfDgQZcxBQUFSkpKkt1ul91uV1JSkgoLC13GHDhwQP369ZO/v79CQkKUmpqq0tLSqu4SAACwqCoXnZMnT6p169aaOXPmecf07t1bubm55vTpp5+6LB81apQWLlyojIwMrV27VsXFxUpISFB5ebk5JjExUdnZ2crMzFRmZqays7OVlJRkLi8vL1ffvn118uRJrV27VhkZGfrggw80ZsyYqu4SAACwqFpVXaFPnz7q06fPBcf4+voqPDy80mUOh0Nz5szR/Pnz1atXL0nSggULFBUVpeXLlys+Pl47d+5UZmamNmzYoI4dO0qSZs+erdjYWO3atUvR0dFaunSpduzYoZycHEVGRkqSpk2bpuTkZL388ssKDAys6q4BAACLqZZrdFauXKnQ0FA1a9ZMKSkpys/PN5dlZWWprKxMcXFx5rzIyEjFxMRo3bp1kqT169fLbrebJUeSOnXqJLvd7jImJibGLDmSFB8fL6fTqaysrEpzOZ1OFRUVuUwAAMC63F50+vTpo/T0dK1YsULTpk3T5s2b1aNHDzmdTklSXl6efHx8FBQU5LJeWFiY8vLyzDGhoaEVth0aGuoyJiwszGV5UFCQfHx8zDHnmjRpknnNj91uV1RU1BXvLwAAuHZV+a2ri3nooYfMn2NiYtS+fXs1atRIixcv1sCBA8+7nmEYstls5uOzf76SMWcbP368Ro8ebT4uKiqi7AAAYGHVfnt5RESEGjVqpN27d0uSwsPDVVpaqoKCApdx+fn55hma8PBwHTlypMK2jh496jLm3DM3BQUFKisrq3Cm5wxfX18FBga6TAAAwLqqvegcO3ZMOTk5ioiIkCS1a9dOtWvX1rJly8wxubm52rZtmzp37ixJio2NlcPh0KZNm8wxGzdulMPhcBmzbds25ebmmmOWLl0qX19ftWvXrrp3CwAAeIAqv3VVXFysPXv2mI/37t2r7OxsBQcHKzg4WGlpabr//vsVERGhffv26YUXXlBISIjuu+8+SZLdbtewYcM0ZswY1a9fX8HBwRo7dqxatWpl3oXVvHlz9e7dWykpKZo1a5Ykafjw4UpISFB0dLQkKS4uTi1atFBSUpKmTJmi48ePa+zYsUpJSeFMDQAAkHQZReerr75S9+7dzcdnrnkZOnSo3njjDW3dulVvv/22CgsLFRERoe7du+u9995TQECAuc706dNVq1YtDRo0SCUlJerZs6fmzZsnb29vc0x6erpSU1PNu7P69+/v8tk93t7eWrx4sZ588kl16dJFfn5+SkxM1NSpU6t+FAAAgCXZDMMwajpETSkqKpLdbpfD4eAsEHAdazxusdu3uW9yX7dvE8BpVfn9zXddAQAAy6LoAAAAy6LoAAAAy6LoAAAAy6LoAAAAy6LoAAAAy6LoAAAAy6LoAAAAy6LoAAAAy6LoAAAAy6LoAAAAy6LoAAAAy6LoAAAAy6LoAAAAy6LoAAAAy6LoAAAAy6LoAAAAy6LoAAAAy6LoAAAAy6LoAAAAy6LoAAAAy6LoAAAAy6LoAAAAy6LoAAAAy6LoAAAAy6LoAAAAy6LoAAAAy6LoAAAAy6LoAAAAy6LoAAAAy6LoAAAAy6LoAAAAy6LoAAAAy6LoAAAAy6LoAAAAy6LoAAAAy6LoAAAAy6LoAAAAy6LoAAAAy6LoAAAAy6py0Vm9erX69eunyMhI2Ww2LVq0yFxWVlam559/Xq1atZK/v78iIyP18MMP6/Dhwy7b6Natm2w2m8s0ePBglzEFBQVKSkqS3W6X3W5XUlKSCgsLXcYcOHBA/fr1k7+/v0JCQpSamqrS0tKq7hIAALCoKhedkydPqnXr1po5c2aFZT/99JO2bNmiP/zhD9qyZYs+/PBDff/99+rfv3+FsSkpKcrNzTWnWbNmuSxPTExUdna2MjMzlZmZqezsbCUlJZnLy8vL1bdvX508eVJr165VRkaGPvjgA40ZM6aquwQAACyqVlVX6NOnj/r06VPpMrvdrmXLlrnMmzFjhjp06KADBw6oYcOG5vy6desqPDy80u3s3LlTmZmZ2rBhgzp27ChJmj17tmJjY7Vr1y5FR0dr6dKl2rFjh3JychQZGSlJmjZtmpKTk/Xyyy8rMDCwqrsGAAAsptqv0XE4HLLZbKpXr57L/PT0dIWEhKhly5YaO3asTpw4YS5bv3697Ha7WXIkqVOnTrLb7Vq3bp05JiYmxiw5khQfHy+n06msrKxKszidThUVFblMAADAuqp8Rqcqfv75Z40bN06JiYkuZ1iGDBmiJk2aKDw8XNu2bdP48eP1zTffmGeD8vLyFBoaWmF7oaGhysvLM8eEhYW5LA8KCpKPj4855lyTJk3Siy++6K7dAwAA17hqKzplZWUaPHiwTp06pddff91lWUpKivlzTEyMmjZtqvbt22vLli1q27atJMlms1XYpmEYLvMvZczZxo8fr9GjR5uPi4qKFBUVVbUdAwAAHqNa3roqKyvToEGDtHfvXi1btuyi18u0bdtWtWvX1u7duyVJ4eHhOnLkSIVxR48eNc/ihIeHVzhzU1BQoLKysgpnes7w9fVVYGCgywQAAKzL7UXnTMnZvXu3li9frvr16190ne3bt6usrEwRERGSpNjYWDkcDm3atMkcs3HjRjkcDnXu3Nkcs23bNuXm5ppjli5dKl9fX7Vr187NewUAADxRld+6Ki4u1p49e8zHe/fuVXZ2toKDgxUZGakHHnhAW7Zs0SeffKLy8nLzrEtwcLB8fHz0ww8/KD09Xffcc49CQkK0Y8cOjRkzRm3atFGXLl0kSc2bN1fv3r2VkpJi3nY+fPhwJSQkKDo6WpIUFxenFi1aKCkpSVOmTNHx48c1duxYpaSkcKYGAABIuowzOl999ZXatGmjNm3aSJJGjx6tNm3a6P/9v/+ngwcP6qOPPtLBgwd12223KSIiwpzO3C3l4+Ojzz//XPHx8YqOjlZqaqri4uK0fPlyeXt7m8+Tnp6uVq1aKS4uTnFxcbr11ls1f/58c7m3t7cWL16sOnXqqEuXLho0aJDuvfdeTZ069UqPCQAAsAibYRhGTYeoKUVFRbLb7XI4HJwFAq5jjcctdvs2903u6/ZtAjitKr+/+a4rAABgWRQdAABgWRQdAABgWRQdAABgWRQdAABgWRQdAABgWRQdAABgWRQdAABgWRQdAABgWRQdAABgWRQdAABgWRQdAABgWRQdAABgWRQdAABgWRQdAABgWRQdAABgWRQdAABgWRQdAABgWRQdAABgWRQdAABgWRQdAABgWRQdAABgWRQdAABgWRQdAABgWRQdAABgWRQdAABgWRQdAABgWRQdAABgWRQdAABgWRQdAABgWRQdAABgWRQdAABgWRQdAABgWRQdAABgWRQdAABgWRQdAABgWRQdAABgWRQdAABgWRQdAABgWRQdAABgWVUuOqtXr1a/fv0UGRkpm82mRYsWuSw3DENpaWmKjIyUn5+funXrpu3bt7uMcTqdGjlypEJCQuTv76/+/fvr4MGDLmMKCgqUlJQku90uu92upKQkFRYWuow5cOCA+vXrJ39/f4WEhCg1NVWlpaVV3SUAAGBRVS46J0+eVOvWrTVz5sxKl7/yyit69dVXNXPmTG3evFnh4eG6++67deLECXPMqFGjtHDhQmVkZGjt2rUqLi5WQkKCysvLzTGJiYnKzs5WZmamMjMzlZ2draSkJHN5eXm5+vbtq5MnT2rt2rXKyMjQBx98oDFjxlR1lwAAgEXZDMMwLntlm00LFy7UvffeK+n02ZzIyEiNGjVKzz//vKTTZ2/CwsL0v//7v3rsscfkcDh04403av78+XrooYckSYcPH1ZUVJQ+/fRTxcfHa+fOnWrRooU2bNigjh07SpI2bNig2NhYfffdd4qOjtZnn32mhIQE5eTkKDIyUpKUkZGh5ORk5efnKzAw8KL5i4qKZLfb5XA4Lmk8AGtqPG6x27e5b3Jft28TwGlV+f3t1mt09u7dq7y8PMXFxZnzfH191bVrV61bt06SlJWVpbKyMpcxkZGRiomJMcesX79edrvdLDmS1KlTJ9ntdpcxMTExZsmRpPj4eDmdTmVlZVWaz+l0qqioyGUCAADW5daik5eXJ0kKCwtzmR8WFmYuy8vLk4+Pj4KCgi44JjQ0tML2Q0NDXcac+zxBQUHy8fExx5xr0qRJ5jU/drtdUVFRl7GXAADAU1TLXVc2m83lsWEYFead69wxlY2/nDFnGz9+vBwOhznl5ORcMBMAAPBsbi064eHhklThjEp+fr559iU8PFylpaUqKCi44JgjR45U2P7Ro0ddxpz7PAUFBSorK6twpucMX19fBQYGukwAAMC63Fp0mjRpovDwcC1btsycV1paqlWrVqlz586SpHbt2ql27douY3Jzc7Vt2zZzTGxsrBwOhzZt2mSO2bhxoxwOh8uYbdu2KTc31xyzdOlS+fr6ql27du7cLQAA4KFqVXWF4uJi7dmzx3y8d+9eZWdnKzg4WA0bNtSoUaM0ceJENW3aVE2bNtXEiRNVt25dJSYmSpLsdruGDRumMWPGqH79+goODtbYsWPVqlUr9erVS5LUvHlz9e7dWykpKZo1a5Ykafjw4UpISFB0dLQkKS4uTi1atFBSUpKmTJmi48ePa+zYsUpJSeFMDQDA47n7bsDr9U7AKhedr776St27dzcfjx49WpI0dOhQzZs3T88995xKSkr05JNPqqCgQB07dtTSpUsVEBBgrjN9+nTVqlVLgwYNUklJiXr27Kl58+bJ29vbHJOenq7U1FTz7qz+/fu7fHaPt7e3Fi9erCeffFJdunSRn5+fEhMTNXXq1KofBQAAYElX9Dk6no7P0QEg8Tk6uDZxRuf8auxzdAAAAK4lFB0AAGBZFB0AAGBZFB0AAGBZFB0AAGBZFB0AAGBZFB0AAGBZFB0AAGBZFB0AAGBZFB0AAGBZFB0AAGBZFB0AAGBZVf72cgAAzocvSMW1hjM6AADAsig6AADAsig6AADAsig6AADAsig6AADAsig6AADAsig6AADAsig6AADAsig6AADAsig6AADAsig6AADAsig6AADAsig6AADAsig6AADAsig6AADAsig6AADAsig6AADAsig6AADAsig6AADAsig6AADAsig6AADAsig6AADAsig6AADAsig6AADAsig6AADAsig6AADAsig6AADAstxedBo3biybzVZheuqppyRJycnJFZZ16tTJZRtOp1MjR45USEiI/P391b9/fx08eNBlTEFBgZKSkmS322W325WUlKTCwkJ37w4AAPBgbi86mzdvVm5urjktW7ZMkvTggw+aY3r37u0y5tNPP3XZxqhRo7Rw4UJlZGRo7dq1Ki4uVkJCgsrLy80xiYmJys7OVmZmpjIzM5Wdna2kpCR37w4AAPBgtdy9wRtvvNHl8eTJk3XzzTera9eu5jxfX1+Fh4dXur7D4dCcOXM0f/589erVS5K0YMECRUVFafny5YqPj9fOnTuVmZmpDRs2qGPHjpKk2bNnKzY2Vrt27VJ0dLS7dwsAAHigar1Gp7S0VAsWLNAjjzwim81mzl+5cqVCQ0PVrFkzpaSkKD8/31yWlZWlsrIyxcXFmfMiIyMVExOjdevWSZLWr18vu91ulhxJ6tSpk+x2uzmmMk6nU0VFRS4TAACwrmotOosWLVJhYaGSk5PNeX369FF6erpWrFihadOmafPmzerRo4ecTqckKS8vTz4+PgoKCnLZVlhYmPLy8swxoaGhFZ4vNDTUHFOZSZMmmdf02O12RUVFuWEvAQDAtcrtb12dbc6cOerTp48iIyPNeQ899JD5c0xMjNq3b69GjRpp8eLFGjhw4Hm3ZRiGy1mhs38+35hzjR8/XqNHjzYfFxUVUXYAALCwais6+/fv1/Lly/Xhhx9ecFxERIQaNWqk3bt3S5LCw8NVWlqqgoICl7M6+fn56ty5sznmyJEjFbZ19OhRhYWFnfe5fH195evrezm7AwAAPFC1vXU1d+5chYaGqm/fvhccd+zYMeXk5CgiIkKS1K5dO9WuXdu8W0uScnNztW3bNrPoxMbGyuFwaNOmTeaYjRs3yuFw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/vnn6tixo/l4/fr1uvnmm2silotrMSOfo3MFTp06pSVLlmjDhg0uX14WGxuruLi4a6Jde0JGuMe3336rrKws/fa3v610+fbt2/XPf/5TEyZMuMrJ/ssTMkrS8ePH5eXl5XKG5GyfffaZ/Pz81K1bt6ua62yekPFcpaWlGjdunL744gt9+OGH530ruKZ5Ss6zffLJJ6pdu7bi4+NrOsp51VRGig4AALAs/jl/Gar6YVw1cd0BGd3HE3KS0X08IScZ3ccTcpLxylB0LsPtt9+ulJQUbdq06bxjHA6HZs+erZiYGH344YdXMd1pZHQfT8hJRvfxhJxkdB9PyEnGK8PFyJdh586dmjhxonr37q3atWurffv2ioyMVJ06dVRQUKAdO3Zo+/btat++vaZMmaI+ffqQ0UMzekpOMl5fOcl4feUk45XhGp0r8PPPP+vTTz/VmjVrtG/fPvNza9q0aaP4+HjFxMTUdEQyupEn5CSj+3hCTjK6jyfkJOPloegAAADL4hodAABgWRQdAABgWRQdAABgWRQdAABgWRQdAABgWRQdoIYlJyfr3nvvrekY15V58+ad9zui3CktLU233XZbtT+P1fF3BFeC28uBGuZwOGQYxlX5xXs9aty4sUaNGqVRo0aZ80pKSnTixAmFhoZW63MXFxfL6XSqfv361fo816Lk5GQVFhZq0aJF19S2cP3hk5GBy1RaWiofH58r3o7dbndDGs9WVlam2rVrX/J4wzBUXl6uWrUu739hfn5+8vPzu6x1q+KGG27QDTfcUO3PUxl3vT4BT8dbV8B/dOvWTSNGjNCIESNUr1491a9fX7///e915qRn48aN9ac//UnJycmy2+1KSUmRJK1bt0533XWX/Pz8FBUVpdTUVJ08eVKSNH78eHXq1KnCc916662aMGGCpIqn5Z1Op1JTUxUaGqo6derojjvu0ObNm83llb3tsmjRItlsNvPxN998o+7duysgIECBgYFq166dvvrqqwvu/8mTJxUYGKh//vOfLvM//vhj+fv768SJE5JOfxnfQw89pKCgINWvX18DBgzQvn37zPGbN2/W3XffrZCQENntdnXt2lVbtmxx2abNZtNf//pXDRgwQP7+/vrTn/50wWwrV66UzWbTkiVL1L59e/n6+mrNmjX64YcfNGDAAIWFhemGG27Q7bffruXLl5vrdevWTfv379czzzwjm81mHqNzj+GZt5jmz5+vxo0by263a/DgweY+S9KJEyc0ZMgQ+fv7KyIiQtOnT1e3bt1czhSd69y3rlauXKkOHTrI399f9erVU5cuXbR///4L7vvZ25k1a5aioqJUt25dPfjggyosLDTHnHkdTZo0SZGRkWrWrJkkaevWrerRo4f8/PxUv359DR8+XMXFxRXWmzhxosLCwlSvXj29+OKL+uWXX/Tss88qODhYDRo00JtvvumS6ULbTUtL01tvvaV//etf5nFfuXKlpIu/fsrLyzV69Gjz7+Bzzz0n3njAlaDoAGd56623VKtWLW3cuFF/+ctfNH36dP397383l0+ZMkUxMTHKysrSH/7wB23dulXx8fEaOHCgvv32W7333ntau3atRowYIUkaMmSINm7cqB9++MHcxvbt27V161YNGTKk0gzPPfecPvjgA7311lvasmWLbrnlFsXHx+v48eOXvB9DhgxRgwYNtHnzZmVlZWncuHEXPWPi7++vwYMHa+7cuS7z586dqwceeEABAQH66aef1L17d91www1avXq11q5dqxtuuEG9e/dWaWmppNOFYOjQoVqzZo02bNigpk2b6p577nEpDZI0YcIEDRgwQFu3btUjjzxySfv13HPPadKkSdq5c6duvfVWFRcX65577tHy5cv19ddfKz4+Xv369TO/SfnDDz9UgwYN9NJLLyk3N1e5ubnn3fYPP/ygRYsW6ZNPPtEnn3yiVatWafLkyeby0aNH68svv9RHH32kZcuWac2aNRUK3IX88ssvuvfee9W1a1d9++23Wr9+vYYPH+5SUC9kz549+sc//qGPP/5YmZmZys7O1lNPPeUy5vPPP9fOnTu1bNkyffLJJ/rpp5/Uu3dvBQUFafPmzXr//fe1fPly8/V5xooVK3T48GGtXr1ar776qtLS0pSQkKCgoCBt3LhRjz/+uB5//HHl5ORI0kW3O3bsWA0aNEi9e/c2j3vnzp0v6fUzbdo0vfnmm5ozZ47Wrl2r48ePa+HChZd8nIEKDACGYRhG165djebNmxunTp0y5z3//PNG8+bNDcMwjEaNGhn33nuvyzpJSUnG8OHDXeatWbPG8PLyMkpKSgzDMIxbb73VeOmll8zl48ePN26//Xbz8dChQ40BAwYYhmEYxcXFRu3atY309HRzeWlpqREZGWm88sorhmEYxty5cw273e7ynAsXLjTO/uscEBBgzJs3r6qHwNi4caPh7e1tHDp0yDAMwzh69KhRu3ZtY+XKlYZhGMacOXOM6Ohol2PkdDoNPz8/Y8mSJZVu85dffjECAgKMjz/+2JwnyRg1atQl5/riiy8MScaiRYsuOrZFixbGjBkzzMeNGjUypk+f7jLm3GM4YcIEo27dukZRUZE579lnnzU6duxoGIZhFBUVGbVr1zbef/99c3lhYaFRt25d4+mnnz5vlgkTJhitW7c2DMMwjh07Zkgyj2VVTJgwwfD29jZycnLMeZ999pnh5eVl5ObmGoZx+nUUFhZmOJ1Oc8zf/vY3IygoyCguLjbnLV682PDy8jLy8vLM9Ro1amSUl5ebY6Kjo40777zTfPzLL78Y/v7+xrvvvlul7Z55XZ9xKa+fiIgIY/LkyebysrIyo0GDBhW2BVwqzugAZ+nUqZPLv7BjY2O1e/dulZeXS5Lat2/vMj4rK0vz5s0zr8W44YYbFB8fr1OnTmnv3r2STp9dSU9Pl3T62pJ33333vGdzfvjhB5WVlalLly7mvNq1a6tDhw7auXPnJe/H6NGj9eijj6pXr16aPHmyyxmlC+nQoYNatmypt99+W5I0f/58NWzYUHfddZe5v3v27FFAQIC5v8HBwfr555/N58jPz9fjjz+uZs2ayW63y263q7i42DzLcsa5x/JSnLvOyZMn9dxzz6lFixaqV6+ebrjhBn333XcVnutSNG7cWAEBAebjiIgI5efnS5L+/e9/q6ysTB06dDCX2+12RUdHX/L2g4ODlZycbJ51+vOf/3zBM0znatiwoRo0aGA+jo2N1alTp7Rr1y5zXqtWrVyuy9m5c6dat24tf39/c16XLl0qrNeyZUt5ef3310FYWJhatWplPvb29lb9+vXN43Gp2z3XxV4/DodDubm5io2NNdepVavWZb1WgDO4GBmogrP/xy5Jp06d0mOPPabU1NQKYxs2bChJSkxM1Lhx47RlyxaVlJQoJydHgwcPrnT7xn+uRTj37QzDMMx5Xl5eFa5ZKCsrc3mclpamxMRELV68WJ999pkmTJigjIwM3XfffRfdx0cffVQzZ87UuHHjNHfuXP32t781n/vUqVNq166dWdzOduONN0o6fc3H0aNH9dprr6lRo0by9fVVbGys+dbEGecey0tx7jrPPvuslixZoqlTp+qWW26Rn5+fHnjggQrPdSnOfWvPZrPp1KlTki7851IVc+fOVWpqqjIzM/Xee+/p97//vZYtW1bpdVwXcybL2ZnOPT5nv27Ot75U+b5f7HhcynbPdSmvH8DdOKMDnGXDhg0VHjdt2lTe3t6Vjm/btq22b9+uW265pcJ05l/WDRo00F133aX09HSlp6erV69eCgsLq3R7Z9Zbu3atOa+srExfffWVmjdvLun0L4QTJ06YFzxLUnZ2doVtNWvWTM8884yWLl2qgQMHVrj25nx+85vf6MCBA/rLX/6i7du3a+jQoS77u3v3boWGhlbY3zN3j61Zs0apqam655571LJlS/n6+urHH3+8pOeuqjVr1ig5OVn33XefWrVqpfDwcJcLWyXJx8fHPCN3uW6++WbVrl1bmzZtMucVFRVp9+7dVd5WmzZtNH78eK1bt04xMTF65513Lmm9AwcO6PDhw+bj9evXy8vLy7zouDItWrRQdna2y2vlyy+/vOh6F3Mp263suF/s9WO32xUREeHy9/CXX35RVlbWZWcFKDrAWXJycjR69Gjt2rVL7777rmbMmKGnn376vOOff/55rV+/Xk899ZSys7O1e/duffTRRxo5cqTLuCFDhigjI0Pvv/++fvOb35x3e/7+/nriiSf07LPPKjMzUzt27FBKSop++uknDRs2TJLUsWNH1a1bVy+88IL27Nmjd955R/PmzTO3UVJSohEjRmjlypXav3+/vvzyS23evNksShcTFBSkgQMH6tlnn1VcXJzL2yVDhgxRSEiIBgwYoDVr1mjv3r1atWqVnn76aR08eFDS6bI2f/587dy5Uxs3btSQIUOq7VbuW265RR9++KGys7P1zTffKDEx0TzrcEbjxo21evVqHTp06LILV0BAgIYOHapnn31WX3zxhbZv365HHnlEXl5el3wx8d69ezV+/HitX79e+/fv19KlS/X9999f8p9LnTp1NHToUH3zzTdmmRw0aJDCw8PPu86QIUPM9bZt26YvvvhCI0eOVFJS0nnL9qW4lO02btxY3377rXbt2qUff/xRZWVll/T6efrppzV58mQtXLhQ3333nZ588kmXu8uAqqLoAGd5+OGHVVJSog4dOuipp57SyJEjNXz48POOv/XWW7Vq1Srt3r1bd955p9q0aaM//OEPioiIcBn34IMP6tixY/rpp58u+gmvkydP1v3336+kpCS1bdtWe/bs0ZIlSxQUFCTp9LUeCxYs0KeffqpWrVrp3XffVVpamrm+t7e3jh07pocffljNmjXToEGD1KdPH7344ouXfByGDRum0tLSCndD1a1bV6tXr1bDhg01cOBANW/eXI888ohKSkoUGBgoSXrzzTdVUFCgNm3aKCkpybxVvjpMnz5dQUFB6ty5s/r166f4+Hi1bdvWZcxLL72kffv26eabb76it0deffVVxcbGKiEhQb169VKXLl3UvHlz1alTxxyTlpamxo0bV7p+3bp19d133+n+++9Xs2bNNHz4cI0YMUKPPfbYJT3/LbfcooEDB+qee+5RXFycYmJi9Prrr19wnbp162rJkiU6fvy4br/9dj3wwAPq2bOnZs6cecn7fbnbTUlJUXR0tNq3b68bb7xRX3755SW9fsaMGaOHH35YycnJio2NVUBAwCW95QqcD5+MDPxHt27ddNttt+m1116r6Sg1Lj09XU8//bQOHz7Mh86dx8mTJ3XTTTdp2rRp5tm25ORkSXI5w+YOaWlpWrRoUaVvUQK4MC5GBmD66aeftHfvXk2aNEmPPfYYJecsX3/9tb777jt16NBBDodDL730kiRpwIAB5phVq1Zp9erVNRURQCV46wq4jvTp08flVvizp4kTJ+qVV17RbbfdprCwMI0fP/6q5Xr88cfPm+vxxx+/ajkuZurUqWrdurV69eqlkydPas2aNQoJCTGX7927V1FRUVXebsuWLc+7/5XdoQTg0vHWFXAdOXTokEpKSipdFhwcrODg4Kuc6LT8/HwVFRVVuiwwMLDav3yzpu3fv7/CRwScERYW5vL5PgCqhqIDAAAsi7euAACAZVF0AACAZVF0AACAZVF0AACAZVF0AACAZVF0AACAZVF0AACAZf1/v1vImpPe0P0AAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# only 16.9% with Employee's KPI above >80 will got promotion.\n",
    "train[['previous_year_rating','is_promoted']].groupby('previous_year_rating').value_counts().plot(kind='bar')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "7c32d7d6-3c5b-468a-bc0f-13043e71c32c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th>is_promoted</th>\n",
       "      <th>0</th>\n",
       "      <th>1</th>\n",
       "      <th>%</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>length_of_service</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>34</th>\n",
       "      <td>3.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>25.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>32</th>\n",
       "      <td>8.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>20.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23</th>\n",
       "      <td>58.0</td>\n",
       "      <td>7.0</td>\n",
       "      <td>10.769231</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>820.0</td>\n",
       "      <td>96.0</td>\n",
       "      <td>10.480349</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>29</th>\n",
       "      <td>27.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>10.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>22</th>\n",
       "      <td>55.0</td>\n",
       "      <td>6.0</td>\n",
       "      <td>9.836066</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>19</th>\n",
       "      <td>297.0</td>\n",
       "      <td>32.0</td>\n",
       "      <td>9.726444</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>2614.0</td>\n",
       "      <td>269.0</td>\n",
       "      <td>9.330558</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>1989.0</td>\n",
       "      <td>204.0</td>\n",
       "      <td>9.302326</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>6089.0</td>\n",
       "      <td>595.0</td>\n",
       "      <td>8.901855</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>6238.0</td>\n",
       "      <td>598.0</td>\n",
       "      <td>8.747806</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>2400.0</td>\n",
       "      <td>229.0</td>\n",
       "      <td>8.710536</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>6424.0</td>\n",
       "      <td>609.0</td>\n",
       "      <td>8.659178</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>4333.0</td>\n",
       "      <td>401.0</td>\n",
       "      <td>8.470638</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>5087.0</td>\n",
       "      <td>464.0</td>\n",
       "      <td>8.358854</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>4170.0</td>\n",
       "      <td>377.0</td>\n",
       "      <td>8.291181</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>5357.0</td>\n",
       "      <td>475.0</td>\n",
       "      <td>8.144719</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>731.0</td>\n",
       "      <td>63.0</td>\n",
       "      <td>7.934509</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13</th>\n",
       "      <td>633.0</td>\n",
       "      <td>54.0</td>\n",
       "      <td>7.860262</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>20</th>\n",
       "      <td>118.0</td>\n",
       "      <td>10.0</td>\n",
       "      <td>7.812500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16</th>\n",
       "      <td>507.0</td>\n",
       "      <td>41.0</td>\n",
       "      <td>7.481752</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15</th>\n",
       "      <td>550.0</td>\n",
       "      <td>43.0</td>\n",
       "      <td>7.251265</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>28</th>\n",
       "      <td>28.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>6.666667</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>18</th>\n",
       "      <td>367.0</td>\n",
       "      <td>25.0</td>\n",
       "      <td>6.377551</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17</th>\n",
       "      <td>406.0</td>\n",
       "      <td>26.0</td>\n",
       "      <td>6.018519</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>14</th>\n",
       "      <td>520.0</td>\n",
       "      <td>29.0</td>\n",
       "      <td>5.282332</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>21</th>\n",
       "      <td>74.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>5.128205</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25</th>\n",
       "      <td>49.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>3.921569</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>27</th>\n",
       "      <td>35.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>2.777778</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>24</th>\n",
       "      <td>70.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>26</th>\n",
       "      <td>41.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>30</th>\n",
       "      <td>12.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>31</th>\n",
       "      <td>20.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>33</th>\n",
       "      <td>9.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>37</th>\n",
       "      <td>1.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "is_promoted             0      1          %\n",
       "length_of_service                          \n",
       "34                    3.0    1.0  25.000000\n",
       "32                    8.0    2.0  20.000000\n",
       "23                   58.0    7.0  10.769231\n",
       "11                  820.0   96.0  10.480349\n",
       "29                   27.0    3.0  10.000000\n",
       "22                   55.0    6.0   9.836066\n",
       "19                  297.0   32.0   9.726444\n",
       "8                  2614.0  269.0   9.330558\n",
       "10                 1989.0  204.0   9.302326\n",
       "2                  6089.0  595.0   8.901855\n",
       "4                  6238.0  598.0   8.747806\n",
       "9                  2400.0  229.0   8.710536\n",
       "3                  6424.0  609.0   8.659178\n",
       "6                  4333.0  401.0   8.470638\n",
       "7                  5087.0  464.0   8.358854\n",
       "1                  4170.0  377.0   8.291181\n",
       "5                  5357.0  475.0   8.144719\n",
       "12                  731.0   63.0   7.934509\n",
       "13                  633.0   54.0   7.860262\n",
       "20                  118.0   10.0   7.812500\n",
       "16                  507.0   41.0   7.481752\n",
       "15                  550.0   43.0   7.251265\n",
       "28                   28.0    2.0   6.666667\n",
       "18                  367.0   25.0   6.377551\n",
       "17                  406.0   26.0   6.018519\n",
       "14                  520.0   29.0   5.282332\n",
       "21                   74.0    4.0   5.128205\n",
       "25                   49.0    2.0   3.921569\n",
       "27                   35.0    1.0   2.777778\n",
       "24                   70.0    NaN        NaN\n",
       "26                   41.0    NaN        NaN\n",
       "30                   12.0    NaN        NaN\n",
       "31                   20.0    NaN        NaN\n",
       "33                    9.0    NaN        NaN\n",
       "37                    1.0    NaN        NaN"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# previous yr rating 5 will getting more promotion\n",
    "exp_tgt = train[['length_of_service','is_promoted']].groupby('length_of_service').value_counts().unstack()\n",
    "exp_tgt['%'] = exp_tgt[1]/(exp_tgt[0]+exp_tgt[1])*100\n",
    "exp_tgt.sort_values(exp_tgt.columns[2], ascending=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c5d2782d-ec6a-4c15-9a81-1cc75f36cedc",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Employee with 34yrs of service got 25% of promotion"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "25fc065d-ccf1-4c0d-9640-a9432b00d2dc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "employee_id               int64\n",
       "department               object\n",
       "region                   object\n",
       "education                object\n",
       "gender                   object\n",
       "recruitment_channel      object\n",
       "no_of_trainings           int64\n",
       "age                       int64\n",
       "previous_year_rating    float64\n",
       "length_of_service         int64\n",
       "KPIs_met >80%             int64\n",
       "awards_won?               int64\n",
       "avg_training_score        int64\n",
       "is_promoted               int64\n",
       "dtype: object"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "378664b5-dbe1-4ad4-9f86-de7192f3b018",
   "metadata": {},
   "outputs": [],
   "source": [
    "cat_cols = ['department','region','education','gender','recruitment_channel']\n",
    "num_cols = ['no_of_trainings','age','previous_year_rating','length_of_service','KPIs_met >80%','awards_won?','avg_training_score']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "ef92623e-62a2-4edc-876f-533efd4f0bfa",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['is_promoted']\n",
      "['employee_id']\n",
      "['department', 'region', 'education', 'gender', 'recruitment_channel']\n",
      "['no_of_trainings', 'age', 'previous_year_rating', 'length_of_service', 'KPIs_met >80%', 'awards_won?', 'avg_training_score']\n"
     ]
    }
   ],
   "source": [
    "print(tgt_col, ign_cols, cat_cols, num_cols, sep='\\n')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "96d88890-9f3b-4d6b-9588-adda5db928ca",
   "metadata": {},
   "outputs": [],
   "source": [
    "cat_pipe_encode = Pipeline(\n",
    "steps = [\n",
    "    ('impute_cat', SimpleImputer(strategy='most_frequent')), # missing values\n",
    "    ('ohe',OneHotEncoder(handle_unknown='ignore')) # categetoy encoding\n",
    "])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "1728fda3-02f7-42de-bf7d-9f324d61d8d3",
   "metadata": {},
   "outputs": [],
   "source": [
    "num_pipe_encode = Pipeline(\n",
    "steps = [\n",
    "    ('impute_num', SimpleImputer(strategy='median')), # missing values\n",
    "    ('scale',StandardScaler()) # standard scaler\n",
    "])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "847c9a00-d770-4693-9a3b-63999924fbfe",
   "metadata": {},
   "outputs": [],
   "source": [
    "preprocess = ColumnTransformer(\n",
    "    transformers =[\n",
    "        ('cat_encode',cat_pipe_encode,cat_cols),\n",
    "        ('num_encode',num_pipe_encode,num_cols)\n",
    "    ]\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "bd0e26e1-c665-495d-afed-3ee90b9e1f21",
   "metadata": {},
   "outputs": [],
   "source": [
    "model_pipeline = Pipeline(\n",
    "steps=[\n",
    "    ('preprocess',preprocess),\n",
    "    ('model',LogisticRegression())\n",
    "])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "7918301b-843b-40ad-915a-d4169e102fe7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>department</th>\n",
       "      <th>region</th>\n",
       "      <th>education</th>\n",
       "      <th>gender</th>\n",
       "      <th>recruitment_channel</th>\n",
       "      <th>no_of_trainings</th>\n",
       "      <th>age</th>\n",
       "      <th>previous_year_rating</th>\n",
       "      <th>length_of_service</th>\n",
       "      <th>KPIs_met &gt;80%</th>\n",
       "      <th>awards_won?</th>\n",
       "      <th>avg_training_score</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Sales &amp; Marketing</td>\n",
       "      <td>region_7</td>\n",
       "      <td>Master's &amp; above</td>\n",
       "      <td>f</td>\n",
       "      <td>sourcing</td>\n",
       "      <td>1</td>\n",
       "      <td>35</td>\n",
       "      <td>5.0</td>\n",
       "      <td>8</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>49</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Operations</td>\n",
       "      <td>region_22</td>\n",
       "      <td>Bachelor's</td>\n",
       "      <td>m</td>\n",
       "      <td>other</td>\n",
       "      <td>1</td>\n",
       "      <td>30</td>\n",
       "      <td>5.0</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>60</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          department     region         education gender recruitment_channel  \\\n",
       "0  Sales & Marketing   region_7  Master's & above      f            sourcing   \n",
       "1         Operations  region_22        Bachelor's      m               other   \n",
       "\n",
       "   no_of_trainings  age  previous_year_rating  length_of_service  \\\n",
       "0                1   35                   5.0                  8   \n",
       "1                1   30                   5.0                  4   \n",
       "\n",
       "   KPIs_met >80%  awards_won?  avg_training_score  \n",
       "0              1            0                  49  \n",
       "1              0            0                  60  "
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X = train.drop(columns=ign_cols+tgt_col)\n",
    "X.head(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "22088da3-e3e7-4d47-98d7-0de36d84ce97",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>is_promoted</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   is_promoted\n",
       "0            0\n",
       "1            0"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y = train[tgt_col]\n",
    "y.head(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "47aefd1f-b5a9-4099-9290-0b7719c85a8e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((49327, 12), (5481, 12), (49327, 1), (5481, 1))"
      ]
     },
     "execution_count": 46,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_X, val_X, train_y, val_y = train_test_split(X,y, \n",
    "                                         random_state=42, test_size=0.1)\n",
    "train_X.shape, val_X.shape, train_y.shape, val_y.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "356fd0a8-8f89-498d-b3fc-f39858134db9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>Pipeline(steps=[(&#x27;preprocess&#x27;,\n",
       "                 ColumnTransformer(transformers=[(&#x27;cat_encode&#x27;,\n",
       "                                                  Pipeline(steps=[(&#x27;impute_cat&#x27;,\n",
       "                                                                   SimpleImputer(strategy=&#x27;most_frequent&#x27;)),\n",
       "                                                                  (&#x27;ohe&#x27;,\n",
       "                                                                   OneHotEncoder(handle_unknown=&#x27;ignore&#x27;))]),\n",
       "                                                  [&#x27;department&#x27;, &#x27;region&#x27;,\n",
       "                                                   &#x27;education&#x27;, &#x27;gender&#x27;,\n",
       "                                                   &#x27;recruitment_channel&#x27;]),\n",
       "                                                 (&#x27;num_encode&#x27;,\n",
       "                                                  Pipeline(steps=[(&#x27;impute_num&#x27;,\n",
       "                                                                   SimpleImputer(strategy=&#x27;median&#x27;)),\n",
       "                                                                  (&#x27;scale&#x27;,\n",
       "                                                                   StandardScaler())]),\n",
       "                                                  [&#x27;no_of_trainings&#x27;, &#x27;age&#x27;,\n",
       "                                                   &#x27;previous_year_rating&#x27;,\n",
       "                                                   &#x27;length_of_service&#x27;,\n",
       "                                                   &#x27;KPIs_met &gt;80%&#x27;,\n",
       "                                                   &#x27;awards_won?&#x27;,\n",
       "                                                   &#x27;avg_training_score&#x27;])])),\n",
       "                (&#x27;model&#x27;, LogisticRegression())])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item sk-dashed-wrapped\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" ><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">Pipeline</label><div class=\"sk-toggleable__content\"><pre>Pipeline(steps=[(&#x27;preprocess&#x27;,\n",
       "                 ColumnTransformer(transformers=[(&#x27;cat_encode&#x27;,\n",
       "                                                  Pipeline(steps=[(&#x27;impute_cat&#x27;,\n",
       "                                                                   SimpleImputer(strategy=&#x27;most_frequent&#x27;)),\n",
       "                                                                  (&#x27;ohe&#x27;,\n",
       "                                                                   OneHotEncoder(handle_unknown=&#x27;ignore&#x27;))]),\n",
       "                                                  [&#x27;department&#x27;, &#x27;region&#x27;,\n",
       "                                                   &#x27;education&#x27;, &#x27;gender&#x27;,\n",
       "                                                   &#x27;recruitment_channel&#x27;]),\n",
       "                                                 (&#x27;num_encode&#x27;,\n",
       "                                                  Pipeline(steps=[(&#x27;impute_num&#x27;,\n",
       "                                                                   SimpleImputer(strategy=&#x27;median&#x27;)),\n",
       "                                                                  (&#x27;scale&#x27;,\n",
       "                                                                   StandardScaler())]),\n",
       "                                                  [&#x27;no_of_trainings&#x27;, &#x27;age&#x27;,\n",
       "                                                   &#x27;previous_year_rating&#x27;,\n",
       "                                                   &#x27;length_of_service&#x27;,\n",
       "                                                   &#x27;KPIs_met &gt;80%&#x27;,\n",
       "                                                   &#x27;awards_won?&#x27;,\n",
       "                                                   &#x27;avg_training_score&#x27;])])),\n",
       "                (&#x27;model&#x27;, LogisticRegression())])</pre></div></div></div><div class=\"sk-serial\"><div class=\"sk-item sk-dashed-wrapped\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-2\" type=\"checkbox\" ><label for=\"sk-estimator-id-2\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">preprocess: ColumnTransformer</label><div class=\"sk-toggleable__content\"><pre>ColumnTransformer(transformers=[(&#x27;cat_encode&#x27;,\n",
       "                                 Pipeline(steps=[(&#x27;impute_cat&#x27;,\n",
       "                                                  SimpleImputer(strategy=&#x27;most_frequent&#x27;)),\n",
       "                                                 (&#x27;ohe&#x27;,\n",
       "                                                  OneHotEncoder(handle_unknown=&#x27;ignore&#x27;))]),\n",
       "                                 [&#x27;department&#x27;, &#x27;region&#x27;, &#x27;education&#x27;, &#x27;gender&#x27;,\n",
       "                                  &#x27;recruitment_channel&#x27;]),\n",
       "                                (&#x27;num_encode&#x27;,\n",
       "                                 Pipeline(steps=[(&#x27;impute_num&#x27;,\n",
       "                                                  SimpleImputer(strategy=&#x27;median&#x27;)),\n",
       "                                                 (&#x27;scale&#x27;, StandardScaler())]),\n",
       "                                 [&#x27;no_of_trainings&#x27;, &#x27;age&#x27;,\n",
       "                                  &#x27;previous_year_rating&#x27;, &#x27;length_of_service&#x27;,\n",
       "                                  &#x27;KPIs_met &gt;80%&#x27;, &#x27;awards_won?&#x27;,\n",
       "                                  &#x27;avg_training_score&#x27;])])</pre></div></div></div><div class=\"sk-parallel\"><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-3\" type=\"checkbox\" ><label for=\"sk-estimator-id-3\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">cat_encode</label><div class=\"sk-toggleable__content\"><pre>[&#x27;department&#x27;, &#x27;region&#x27;, &#x27;education&#x27;, &#x27;gender&#x27;, &#x27;recruitment_channel&#x27;]</pre></div></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-4\" type=\"checkbox\" ><label for=\"sk-estimator-id-4\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">SimpleImputer</label><div class=\"sk-toggleable__content\"><pre>SimpleImputer(strategy=&#x27;most_frequent&#x27;)</pre></div></div></div><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-5\" type=\"checkbox\" ><label for=\"sk-estimator-id-5\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">OneHotEncoder</label><div class=\"sk-toggleable__content\"><pre>OneHotEncoder(handle_unknown=&#x27;ignore&#x27;)</pre></div></div></div></div></div></div></div></div><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-6\" type=\"checkbox\" ><label for=\"sk-estimator-id-6\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">num_encode</label><div class=\"sk-toggleable__content\"><pre>[&#x27;no_of_trainings&#x27;, &#x27;age&#x27;, &#x27;previous_year_rating&#x27;, &#x27;length_of_service&#x27;, &#x27;KPIs_met &gt;80%&#x27;, &#x27;awards_won?&#x27;, &#x27;avg_training_score&#x27;]</pre></div></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-7\" type=\"checkbox\" ><label for=\"sk-estimator-id-7\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">SimpleImputer</label><div class=\"sk-toggleable__content\"><pre>SimpleImputer(strategy=&#x27;median&#x27;)</pre></div></div></div><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-8\" type=\"checkbox\" ><label for=\"sk-estimator-id-8\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">StandardScaler</label><div class=\"sk-toggleable__content\"><pre>StandardScaler()</pre></div></div></div></div></div></div></div></div></div></div><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-9\" type=\"checkbox\" ><label for=\"sk-estimator-id-9\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">LogisticRegression</label><div class=\"sk-toggleable__content\"><pre>LogisticRegression()</pre></div></div></div></div></div></div></div>"
      ],
      "text/plain": [
       "Pipeline(steps=[('preprocess',\n",
       "                 ColumnTransformer(transformers=[('cat_encode',\n",
       "                                                  Pipeline(steps=[('impute_cat',\n",
       "                                                                   SimpleImputer(strategy='most_frequent')),\n",
       "                                                                  ('ohe',\n",
       "                                                                   OneHotEncoder(handle_unknown='ignore'))]),\n",
       "                                                  ['department', 'region',\n",
       "                                                   'education', 'gender',\n",
       "                                                   'recruitment_channel']),\n",
       "                                                 ('num_encode',\n",
       "                                                  Pipeline(steps=[('impute_num',\n",
       "                                                                   SimpleImputer(strategy='median')),\n",
       "                                                                  ('scale',\n",
       "                                                                   StandardScaler())]),\n",
       "                                                  ['no_of_trainings', 'age',\n",
       "                                                   'previous_year_rating',\n",
       "                                                   'length_of_service',\n",
       "                                                   'KPIs_met >80%',\n",
       "                                                   'awards_won?',\n",
       "                                                   'avg_training_score'])])),\n",
       "                ('model', LogisticRegression())])"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model_pipeline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "18d2c018-49a6-4e93-b785-33f75127b6dc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-2 {color: black;background-color: white;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-2\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>Pipeline(steps=[(&#x27;preprocess&#x27;,\n",
       "                 ColumnTransformer(transformers=[(&#x27;cat_encode&#x27;,\n",
       "                                                  Pipeline(steps=[(&#x27;impute_cat&#x27;,\n",
       "                                                                   SimpleImputer(strategy=&#x27;most_frequent&#x27;)),\n",
       "                                                                  (&#x27;ohe&#x27;,\n",
       "                                                                   OneHotEncoder(handle_unknown=&#x27;ignore&#x27;))]),\n",
       "                                                  [&#x27;department&#x27;, &#x27;region&#x27;,\n",
       "                                                   &#x27;education&#x27;, &#x27;gender&#x27;,\n",
       "                                                   &#x27;recruitment_channel&#x27;]),\n",
       "                                                 (&#x27;num_encode&#x27;,\n",
       "                                                  Pipeline(steps=[(&#x27;impute_num&#x27;,\n",
       "                                                                   SimpleImputer(strategy=&#x27;median&#x27;)),\n",
       "                                                                  (&#x27;scale&#x27;,\n",
       "                                                                   StandardScaler())]),\n",
       "                                                  [&#x27;no_of_trainings&#x27;, &#x27;age&#x27;,\n",
       "                                                   &#x27;previous_year_rating&#x27;,\n",
       "                                                   &#x27;length_of_service&#x27;,\n",
       "                                                   &#x27;KPIs_met &gt;80%&#x27;,\n",
       "                                                   &#x27;awards_won?&#x27;,\n",
       "                                                   &#x27;avg_training_score&#x27;])])),\n",
       "                (&#x27;model&#x27;, LogisticRegression())])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item sk-dashed-wrapped\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-10\" type=\"checkbox\" ><label for=\"sk-estimator-id-10\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">Pipeline</label><div class=\"sk-toggleable__content\"><pre>Pipeline(steps=[(&#x27;preprocess&#x27;,\n",
       "                 ColumnTransformer(transformers=[(&#x27;cat_encode&#x27;,\n",
       "                                                  Pipeline(steps=[(&#x27;impute_cat&#x27;,\n",
       "                                                                   SimpleImputer(strategy=&#x27;most_frequent&#x27;)),\n",
       "                                                                  (&#x27;ohe&#x27;,\n",
       "                                                                   OneHotEncoder(handle_unknown=&#x27;ignore&#x27;))]),\n",
       "                                                  [&#x27;department&#x27;, &#x27;region&#x27;,\n",
       "                                                   &#x27;education&#x27;, &#x27;gender&#x27;,\n",
       "                                                   &#x27;recruitment_channel&#x27;]),\n",
       "                                                 (&#x27;num_encode&#x27;,\n",
       "                                                  Pipeline(steps=[(&#x27;impute_num&#x27;,\n",
       "                                                                   SimpleImputer(strategy=&#x27;median&#x27;)),\n",
       "                                                                  (&#x27;scale&#x27;,\n",
       "                                                                   StandardScaler())]),\n",
       "                                                  [&#x27;no_of_trainings&#x27;, &#x27;age&#x27;,\n",
       "                                                   &#x27;previous_year_rating&#x27;,\n",
       "                                                   &#x27;length_of_service&#x27;,\n",
       "                                                   &#x27;KPIs_met &gt;80%&#x27;,\n",
       "                                                   &#x27;awards_won?&#x27;,\n",
       "                                                   &#x27;avg_training_score&#x27;])])),\n",
       "                (&#x27;model&#x27;, LogisticRegression())])</pre></div></div></div><div class=\"sk-serial\"><div class=\"sk-item sk-dashed-wrapped\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-11\" type=\"checkbox\" ><label for=\"sk-estimator-id-11\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">preprocess: ColumnTransformer</label><div class=\"sk-toggleable__content\"><pre>ColumnTransformer(transformers=[(&#x27;cat_encode&#x27;,\n",
       "                                 Pipeline(steps=[(&#x27;impute_cat&#x27;,\n",
       "                                                  SimpleImputer(strategy=&#x27;most_frequent&#x27;)),\n",
       "                                                 (&#x27;ohe&#x27;,\n",
       "                                                  OneHotEncoder(handle_unknown=&#x27;ignore&#x27;))]),\n",
       "                                 [&#x27;department&#x27;, &#x27;region&#x27;, &#x27;education&#x27;, &#x27;gender&#x27;,\n",
       "                                  &#x27;recruitment_channel&#x27;]),\n",
       "                                (&#x27;num_encode&#x27;,\n",
       "                                 Pipeline(steps=[(&#x27;impute_num&#x27;,\n",
       "                                                  SimpleImputer(strategy=&#x27;median&#x27;)),\n",
       "                                                 (&#x27;scale&#x27;, StandardScaler())]),\n",
       "                                 [&#x27;no_of_trainings&#x27;, &#x27;age&#x27;,\n",
       "                                  &#x27;previous_year_rating&#x27;, &#x27;length_of_service&#x27;,\n",
       "                                  &#x27;KPIs_met &gt;80%&#x27;, &#x27;awards_won?&#x27;,\n",
       "                                  &#x27;avg_training_score&#x27;])])</pre></div></div></div><div class=\"sk-parallel\"><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-12\" type=\"checkbox\" ><label for=\"sk-estimator-id-12\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">cat_encode</label><div class=\"sk-toggleable__content\"><pre>[&#x27;department&#x27;, &#x27;region&#x27;, &#x27;education&#x27;, &#x27;gender&#x27;, &#x27;recruitment_channel&#x27;]</pre></div></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-13\" type=\"checkbox\" ><label for=\"sk-estimator-id-13\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">SimpleImputer</label><div class=\"sk-toggleable__content\"><pre>SimpleImputer(strategy=&#x27;most_frequent&#x27;)</pre></div></div></div><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-14\" type=\"checkbox\" ><label for=\"sk-estimator-id-14\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">OneHotEncoder</label><div class=\"sk-toggleable__content\"><pre>OneHotEncoder(handle_unknown=&#x27;ignore&#x27;)</pre></div></div></div></div></div></div></div></div><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-15\" type=\"checkbox\" ><label for=\"sk-estimator-id-15\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">num_encode</label><div class=\"sk-toggleable__content\"><pre>[&#x27;no_of_trainings&#x27;, &#x27;age&#x27;, &#x27;previous_year_rating&#x27;, &#x27;length_of_service&#x27;, &#x27;KPIs_met &gt;80%&#x27;, &#x27;awards_won?&#x27;, &#x27;avg_training_score&#x27;]</pre></div></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-16\" type=\"checkbox\" ><label for=\"sk-estimator-id-16\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">SimpleImputer</label><div class=\"sk-toggleable__content\"><pre>SimpleImputer(strategy=&#x27;median&#x27;)</pre></div></div></div><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-17\" type=\"checkbox\" ><label for=\"sk-estimator-id-17\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">StandardScaler</label><div class=\"sk-toggleable__content\"><pre>StandardScaler()</pre></div></div></div></div></div></div></div></div></div></div><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-18\" type=\"checkbox\" ><label for=\"sk-estimator-id-18\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">LogisticRegression</label><div class=\"sk-toggleable__content\"><pre>LogisticRegression()</pre></div></div></div></div></div></div></div>"
      ],
      "text/plain": [
       "Pipeline(steps=[('preprocess',\n",
       "                 ColumnTransformer(transformers=[('cat_encode',\n",
       "                                                  Pipeline(steps=[('impute_cat',\n",
       "                                                                   SimpleImputer(strategy='most_frequent')),\n",
       "                                                                  ('ohe',\n",
       "                                                                   OneHotEncoder(handle_unknown='ignore'))]),\n",
       "                                                  ['department', 'region',\n",
       "                                                   'education', 'gender',\n",
       "                                                   'recruitment_channel']),\n",
       "                                                 ('num_encode',\n",
       "                                                  Pipeline(steps=[('impute_num',\n",
       "                                                                   SimpleImputer(strategy='median')),\n",
       "                                                                  ('scale',\n",
       "                                                                   StandardScaler())]),\n",
       "                                                  ['no_of_trainings', 'age',\n",
       "                                                   'previous_year_rating',\n",
       "                                                   'length_of_service',\n",
       "                                                   'KPIs_met >80%',\n",
       "                                                   'awards_won?',\n",
       "                                                   'avg_training_score'])])),\n",
       "                ('model', LogisticRegression())])"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model_pipeline.fit(train_X, train_y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "8ebe61a5-51a3-49a1-ab28-095cb7a79a69",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0.99451187, 0.00548813],\n",
       "       [0.98246063, 0.01753937],\n",
       "       [0.95742303, 0.04257697],\n",
       "       ...,\n",
       "       [0.93959239, 0.06040761],\n",
       "       [0.95290112, 0.04709888],\n",
       "       [0.97669398, 0.02330602]])"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model_pipeline.predict_proba(val_X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "624e825c-a5eb-4e27-98be-7b6f91ee02ff",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.99451187, 0.98246063, 0.95742303, ..., 0.93959239, 0.95290112,\n",
       "       0.97669398])"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model_pipeline.predict_proba(val_X)[:,0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "b842933b-bc16-4163-9d40-7c36d1d0c330",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.00548813, 0.01753937, 0.04257697, ..., 0.06040761, 0.04709888,\n",
       "       0.02330602])"
      ]
     },
     "execution_count": 51,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model_pipeline.predict_proba(val_X)[:,1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "9edac5fc-69cd-484a-97df-26983ed1881c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 0, 0, ..., 0, 0, 0], dtype=int64)"
      ]
     },
     "execution_count": 52,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model_pipeline.predict(val_X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "243d6142-5de5-4aa8-a010-025ce7988f9e",
   "metadata": {},
   "outputs": [],
   "source": [
    "def model_train_val_eval(train_X,val_X,train_y,val_y,model_pipeline):\n",
    "    \n",
    "    predicted_train_tgt = model_pipeline.predict(train_X)\n",
    "    predicted_val_tgt = model_pipeline.predict(val_X)\n",
    "\n",
    "    print('Train AUC', roc_auc_score(train_y,predicted_train_tgt),sep='\\n')\n",
    "    print('Valid AUC', roc_auc_score(val_y,predicted_val_tgt),sep='\\n')\n",
    "\n",
    "    print('Train cnf_matrix', confusion_matrix(train_y,predicted_train_tgt),sep='\\n')\n",
    "    print('Valid cnf_matrix', confusion_matrix(val_y,predicted_val_tgt),sep='\\n')\n",
    "\n",
    "    print('Train cls_rep', classification_report(train_y,predicted_train_tgt),sep='\\n')\n",
    "    print('Valid cls rep', classification_report(val_y,predicted_val_tgt),sep='\\n')\n",
    "\n",
    "    y_pred_proba = model_pipeline.predict_proba(val_X)[:,1]\n",
    "    plt.figure()\n",
    "    fpr, tpr, thrsh = roc_curve(val_y,y_pred_proba)\n",
    "    #roc_auc = auc(fpr, tpr)\n",
    "    \n",
    "    plt.plot(fpr, tpr)\n",
    "    plt.show()\n",
    "  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "8aa3f35d-aa96-4a9b-9f55-1b1a2de6ba6e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train AUC\n",
      "0.6289025442153221\n",
      "Valid AUC\n",
      "0.6303413659231352\n",
      "Train cnf_matrix\n",
      "[[44838   252]\n",
      " [ 3121  1116]]\n",
      "Valid cnf_matrix\n",
      "[[5019   31]\n",
      " [ 316  115]]\n",
      "Train cls_rep\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.93      0.99      0.96     45090\n",
      "           1       0.82      0.26      0.40      4237\n",
      "\n",
      "    accuracy                           0.93     49327\n",
      "   macro avg       0.88      0.63      0.68     49327\n",
      "weighted avg       0.92      0.93      0.92     49327\n",
      "\n",
      "Valid cls rep\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.94      0.99      0.97      5050\n",
      "           1       0.79      0.27      0.40       431\n",
      "\n",
      "    accuracy                           0.94      5481\n",
      "   macro avg       0.86      0.63      0.68      5481\n",
      "weighted avg       0.93      0.94      0.92      5481\n",
      "\n"
     ]
    },
    {
     "data": {
      "image/png": "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wkiS862eGRlEWDkDAIgnwkiSC149QxABAMQbYQR+799/OUEEABB3hBH4MUUEAGAHwkgS8+0/AwCAnaholaSMMbqxtkktX35rd1MAAEmOnpEkdez4UEAQKS44lboiAABb0DOSpEaWfW9+dLGmZ7uoKwIAsAU9I0kouOw7G+EBAOwUURipqalRYWGhMjMzVVRUpMbGxnHP7+/v15o1a1RQUCC3261zzjlH27Zti6jBmBzKvgMAphrLwzR1dXVatWqVampqtHDhQr3wwgtaunSp2tradPbZZ4e85uabb9bXX3+trVu36uc//7mOHDmiwcHBSTce1oSqtkrZdwCA3VKMGTl7YGILFizQ/PnztWXLFv+xOXPmaPny5aqurh51/gcffKBf/epX2r9/v0477bSIGun1epWTk6Oenh55PJ6IfkayM8ao++iAip/4f5KotgoAiL1wn9+WhmkGBgbU0tKisrKygONlZWXas2dPyGvee+89FRcX66mnntJZZ52l8847Tw8++KCOHTs25uf09/fL6/UGvBAZY4yO9g/quo27/UFEotoqAGDqsDRM09XVpaGhIeXm5gYcz83NVWdnZ8hr9u/fr927dyszM1PvvPOOurq6dM899+ibb74Zc95IdXW11q1bZ6VpCME3UdW394xPccGpynIxTwQAMDVEtLQ3eI6BMWbMeQfDw8NKSUnR9u3blZOTI0nasGGDbrzxRm3evFnTpk0bdU1VVZUqKyv9771er/Lz8yNpatIKnqgqnZisuqOihNUzAIApxVIYmTFjhtLS0kb1ghw5cmRUb4lPXl6ezjrrLH8QkU7MMTHG6NChQzr33HNHXeN2u+V2u600DSMEBxHfRFVCCABgKrI0Z8TlcqmoqEgNDQ0BxxsaGlRaWhrymoULF+rw4cP6/vvv/cc+++wzpaamatasWRE0GeMJtWLmo8orlO1OJ4gAAKYky3VGKisr9fLLL2vbtm3at2+fVq9erfb2dlVUVEg6McSyYsUK//m33HKLpk+frjvuuENtbW36+OOP9dBDD+k3v/lNyCEaTE7fwJB/jggrZgAATmB5zkh5ebm6u7u1fv16dXR0aN68eaqvr1dBQYEkqaOjQ+3t7f7zf/KTn6ihoUH333+/iouLNX36dN1888164oknovctIGl0ZVVWzAAAnMBynRE7UGdkYqEqq/7fSgqaAQDsE5M6I5iaxpqwShABADgBYcThQgUR5okAAJyEMOJgY62cIYgAAJyEMOJgx46zcgYA4HyEkQTByhkAgFMRRhxs5Doo5qoCAJyKMOJQxhjdVNtkdzMAAJg0wohDjay0OjfPo2kZ7MILAHAmwogDBVda3VFRQk0RAIBjEUYcJng579w8j7Jc9IoAAJyLMOIwwRvhUWkVAOB0hBEHCZ60ynJeAEAiIIw4SPCkVYZnAACJgDDiEMG9IkxaBQAkCsKIQ9ArAgBIVIQRB6BXBACQyAgjDkCvCAAgkRFGpjgKnAEAEl263Q1AaMYY9Q0MUeAMAJDwCCNTkDFGN9Y2qeXLb/3HKHAGAEhUDNNMQX0DQwFBZG6eRx9VXkGBMwBAQqJnZIoJXjnT/OhiTc920SMCAEhYhJEpwhijY8eHRq2cIYgAABIdYWQKCDVHRGLlDAAgOTBnZAoIniMiScUFp7JyBgCQFOgZsdHI5bs+zY8uVpYrTdMy0ugVAQAkBcKIDXwh5KbaJv/8EIk5IgCA5EQYibOx5ofMzfNQRwQAkJQII3F27PjoGiI7KkqU5WJYBgCQnAgjNqKGCAAArKaJO2NO/pneEAAACCNxFVxdFQAAEEbiKri66rQM6ogAAEAYiZPhYRNQT4TqqgAAnEAYiQNjTgSRA11HJZ3oFaG6KgAAJxBG4uDY8ZPDM4UzsqknAgDACISROHv//suVmkoQAQDAhzASY77S7z50iAAAEIiiZzE0Vul3AABwEj0jMdQ3EFj6vbjgVJbzAgAQhJ6RGAleykvpdwAAQqNnJAaGh42u3rArYCkvQQQAgNAII1EWHERYygsAwPgII1EUKoh8VHkFS3kBABgHYSRKCCIAAESGMBIFBBEAACJHGJkkgggAAJNDGJmE4A3wCCIAAFhHGJmE4A3wCCIAAFhHGJkEY07+mQ3wAACIDGEkQsYY3VTb5H9PGREAACJDGIlQ38DJIZq5eR72nAEAIEKEkQgE94rsqCihwioAABEijERg5MTVuXkeZbnoFQEAIFKEkQiMnLhKrwgAAJNDGLGIiasAAERXRGGkpqZGhYWFyszMVFFRkRobG8O67pNPPlF6erouueSSSD52SggeomHiKgAAk2M5jNTV1WnVqlVas2aNWltbtWjRIi1dulTt7e3jXtfT06MVK1bo6quvjrixUw1DNAAATJ7lMLJhwwbdeeeduuuuuzRnzhw9++yzys/P15YtW8a97u6779Ytt9yikpKSiBs71ZBDAACYPEthZGBgQC0tLSorKws4XlZWpj179ox53SuvvKIvvvhCa9euDetz+vv75fV6A14AACAxWQojXV1dGhoaUm5ubsDx3NxcdXZ2hrzm888/18MPP6zt27crPT09rM+prq5WTk6O/5Wfn2+lmQAAwEEimsAaPE/CGBNy7sTQ0JBuueUWrVu3Tuedd17YP7+qqko9PT3+18GDByNpJgAAcIDwuip+NGPGDKWlpY3qBTly5Mio3hJJ6u3tVXNzs1pbW3XfffdJkoaHh2WMUXp6unbu3Kmrrrpq1HVut1tut9tK0+JmZI0RAAAweZZ6Rlwul4qKitTQ0BBwvKGhQaWlpaPO93g8+vTTT7V3717/q6KiQueff7727t2rBQsWTK71cTY8bPTLTbvtbgYAAAnFUs+IJFVWVurWW29VcXGxSkpK9OKLL6q9vV0VFRWSTgyxfPXVV3rttdeUmpqqefPmBVx/xhlnKDMzc9Txqc6YE0HkQNdRSdQYAQAgWiyHkfLycnV3d2v9+vXq6OjQvHnzVF9fr4KCAklSR0fHhDVHnGhksbPCGdl6//7LqTECAEAUpBgz9WdBeL1e5eTkqKenRx6Px5Y2HO0f1IVrP5Qk/WfdEmW7Lec4AACSSrjPb/amCQP70QAAEDuEkTCwHw0AALFDGLGI/WgAAIguwsgEjDHqGxjyvyeHAAAQXczCHIcxRjfWNqnly2/tbgoAAAmLnpFxHDs+FBBEigtOZb4IAABRRs9ImJofXazp2S7miwAAEGX0jIQpy5VGEAEAIAYII+OY+uXgAABwPsLIGIILnQEAgNggjIyBQmcAAMQHYSQMFDoDACB2CCNjGDlfhBwCAEDsEEZCYL4IAADxQxgJgfkiAADED2FkAswXAQAgtggjEyCHAAAQW4QRAABgK8JICFReBQAgfggjQVhJAwBAfBFGgrCSBgCA+CKMjIOVNAAAxB5hZBzkEAAAYo8wAgAAbEUYAQAAtiKMAAAAWxFGglBjBACA+CKMjECNEQAA4o8wMkLfADVGAACIN8LIj4J7RagxAgBAfBBGfhRceTXLRa8IAADxQBgJgV4RAADihzASAjkEAID4IYz8iCW9AADYgzAilvQCAGAnwohGT15lSS8AAPFDGAnC5FUAAOKLMKLA+SLkEAAA4ivpwwjzRQAAsFfShxHmiwAAYK+kDyMjMV8EAID4I4yMQA4BACD+CCMAAMBWhBEAAGArwggAALAVYQQAANiKMAIAAGxFGAEAALZK+jAyshQ8AACIv6QOI5SCBwDAfkkdRigFDwCA/ZI6jIxEKXgAAOxBGPkROQQAAHsQRgAAgK0IIwAAwFaEEQAAYKuIwkhNTY0KCwuVmZmpoqIiNTY2jnnu22+/rWuuuUann366PB6PSkpK9OGHH0bcYAAAkFgsh5G6ujqtWrVKa9asUWtrqxYtWqSlS5eqvb095Pkff/yxrrnmGtXX16ulpUVXXnmlli1bptbW1kk3HgAAOF+KMdZqkC5YsEDz58/Xli1b/MfmzJmj5cuXq7q6OqyfceGFF6q8vFyPPfZYWOd7vV7l5OSop6dHHo/HSnPH1TcwqLmPneilaVu/RFmu9Kj9bAAAkl24z29LPSMDAwNqaWlRWVlZwPGysjLt2bMnrJ8xPDys3t5enXbaaWOe09/fL6/XG/ACAACJyVIY6erq0tDQkHJzcwOO5+bmqrOzM6yf8cwzz+jo0aO6+eabxzynurpaOTk5/ld+fr6VZgIAAAeJaAJrcKVSY0xY1UvfeOMNPf7446qrq9MZZ5wx5nlVVVXq6enxvw4ePBhJMwEAgANYmiQxY8YMpaWljeoFOXLkyKjekmB1dXW68847tWPHDi1evHjcc91ut9xut5WmRYQdewEAsJ+lnhGXy6WioiI1NDQEHG9oaFBpaemY173xxhu6/fbb9frrr+u6666LrKVRxo69AABMDZaXj1RWVurWW29VcXGxSkpK9OKLL6q9vV0VFRWSTgyxfPXVV3rttdcknQgiK1as0HPPPafLLrvM36sybdo05eTkRPGrWMOOvQAATA2Ww0h5ebm6u7u1fv16dXR0aN68eaqvr1dBQYEkqaOjI6DmyAsvvKDBwUHde++9uvfee/3Hb7vtNr366quT/wZRwI69AADYx3KdETvEos4INUYAAIitmNQZAQAAiDbCCAAAsBVhBAAA2IowAgAAbEUYAQAAtiKMAAAAWxFGAACArQgjAADAVoQRAABgK8IIAACwFWEEAADYijACAABsRRgBAAC2IowAAABbEUYAAICtCCMAAMBWhBEAAGArwggAALAVYQQAANiKMAIAAGxFGAEAALYijAAAAFsRRgAAgK0IIwAAwFaEEQAAYCvCCAAAsBVhBAAA2IowAgAAbEUYAQAAtiKMAAAAWxFGAACArQgjAADAVoQRAABgK8IIAACwFWEEAADYijACAABsRRgBAAC2IowAAABbEUYAAICtCCMAAMBWhBEAAGArwggAALAVYQQAANiKMAIAAGxFGAEAALYijAAAAFsRRgAAgK0IIwAAwFaEEQAAYCvCCAAAsBVhBAAA2IowAgAAbEUYAQAAtiKMAAAAWxFGAACArQgjAADAVhGFkZqaGhUWFiozM1NFRUVqbGwc9/xdu3apqKhImZmZmj17tmprayNqLAAASDyWw0hdXZ1WrVqlNWvWqLW1VYsWLdLSpUvV3t4e8vwDBw7o2muv1aJFi9Ta2qpHHnlEK1eu1FtvvTXpxgMAAOdLMcYYKxcsWLBA8+fP15YtW/zH5syZo+XLl6u6unrU+X/4wx/03nvvad++ff5jFRUV+ve//62mpqawPtPr9SonJ0c9PT3yeDxWmjumvoFBzX3sQ0lS2/olynKlR+XnAgCAE8J9flvqGRkYGFBLS4vKysoCjpeVlWnPnj0hr2lqahp1/pIlS9Tc3Kzjx4+HvKa/v19erzfgBQAAEpOlMNLV1aWhoSHl5uYGHM/NzVVnZ2fIazo7O0OePzg4qK6urpDXVFdXKycnx//Kz8+30kwAAOAgEU1gTUlJCXhvjBl1bKLzQx33qaqqUk9Pj/918ODBSJo5rmkZaWpbv0Rt65doWkZa1H8+AAAIj6WJEjNmzFBaWtqoXpAjR46M6v3wOfPMM0Oen56erunTp4e8xu12y+12W2maZSkpKcwTAQBgCrDUM+JyuVRUVKSGhoaA4w0NDSotLQ15TUlJyajzd+7cqeLiYmVkZFhsLgAASDSWh2kqKyv18ssva9u2bdq3b59Wr16t9vZ2VVRUSDoxxLJixQr/+RUVFfryyy9VWVmpffv2adu2bdq6dasefPDB6H0LAADgWJbHKcrLy9Xd3a3169ero6ND8+bNU319vQoKCiRJHR0dATVHCgsLVV9fr9WrV2vz5s2aOXOmNm7cqBtuuCF63wIAADiW5TojdohFnREAABBbMakzAgAAEG2EEQAAYCvCCAAAsBVhBAAA2IowAgAAbEUYAQAAtiKMAAAAWxFGAACArQgjAADAVo7YttZXJNbr9drcEgAAEC7fc3uiYu+OCCO9vb2SpPz8fJtbAgAArOrt7VVOTs6Yf++IvWmGh4d1+PBhnXLKKUpJSYnaz/V6vcrPz9fBgwfZ8ybGuNfxwX2OD+5zfHCf4yOW99kYo97eXs2cOVOpqWPPDHFEz0hqaqpmzZoVs5/v8Xj4RY8T7nV8cJ/jg/scH9zn+IjVfR6vR8SHCawAAMBWhBEAAGCrpA4jbrdba9euldvttrspCY97HR/c5/jgPscH9zk+psJ9dsQEVgAAkLiSumcEAADYjzACAABsRRgBAAC2IowAAABbJXwYqampUWFhoTIzM1VUVKTGxsZxz9+1a5eKioqUmZmp2bNnq7a2Nk4tdTYr9/ntt9/WNddco9NPP10ej0clJSX68MMP49haZ7P6O+3zySefKD09XZdccklsG5ggrN7n/v5+rVmzRgUFBXK73TrnnHO0bdu2OLXWuaze5+3bt+viiy9WVlaW8vLydMcdd6i7uztOrXWmjz/+WMuWLdPMmTOVkpKid999d8Jr4v4sNAnsz3/+s8nIyDAvvfSSaWtrMw888IDJzs42X375Zcjz9+/fb7KysswDDzxg2trazEsvvWQyMjLMm2++GeeWO4vV+/zAAw+YJ5980vzzn/80n332mamqqjIZGRnmX//6V5xb7jxW77XPd999Z2bPnm3KysrMxRdfHJ/GOlgk9/n66683CxYsMA0NDebAgQPmH//4h/nkk0/i2GrnsXqfGxsbTWpqqnnuuefM/v37TWNjo7nwwgvN8uXL49xyZ6mvrzdr1qwxb731lpFk3nnnnXHPt+NZmNBh5NJLLzUVFRUBxy644ALz8MMPhzz/97//vbngggsCjt19993msssui1kbE4HV+xzK3Llzzbp166LdtIQT6b0uLy83jz76qFm7di1hJAxW7/Nf//pXk5OTY7q7u+PRvIRh9T7/8Y9/NLNnzw44tnHjRjNr1qyYtTHRhBNG7HgWJuwwzcDAgFpaWlRWVhZwvKysTHv27Al5TVNT06jzlyxZoubmZh0/fjxmbXWySO5zsOHhYfX29uq0006LRRMTRqT3+pVXXtEXX3yhtWvXxrqJCSGS+/zee++puLhYTz31lM466yydd955evDBB3Xs2LF4NNmRIrnPpaWlOnTokOrr62WM0ddff60333xT1113XTyanDTseBY6YqO8SHR1dWloaEi5ubkBx3Nzc9XZ2Rnyms7OzpDnDw4OqqurS3l5eTFrr1NFcp+DPfPMMzp69KhuvvnmWDQxYURyrz///HM9/PDDamxsVHp6wv7nHlWR3Of9+/dr9+7dyszM1DvvvKOuri7dc889+uabb5g3MoZI7nNpaam2b9+u8vJy/fDDDxocHNT111+vTZs2xaPJScOOZ2HC9oz4pKSkBLw3xow6NtH5oY4jkNX77PPGG2/o8ccfV11dnc4444xYNS+hhHuvh4aGdMstt2jdunU677zz4tW8hGHld3p4eFgpKSnavn27Lr30Ul177bXasGGDXn31VXpHJmDlPre1tWnlypV67LHH1NLSog8++EAHDhxQRUVFPJqaVOL9LEzYfyrNmDFDaWlpoxL2kSNHRiU+nzPPPDPk+enp6Zo+fXrM2upkkdxnn7q6Ot15553asWOHFi9eHMtmJgSr97q3t1fNzc1qbW3VfffdJ+nEQ9MYo/T0dO3cuVNXXXVVXNruJJH8Tufl5emss84K2Cp9zpw5Msbo0KFDOvfcc2PaZieK5D5XV1dr4cKFeuihhyRJF110kbKzs7Vo0SI98cQT9F5HiR3PwoTtGXG5XCoqKlJDQ0PA8YaGBpWWloa8pqSkZNT5O3fuVHFxsTIyMmLWVieL5D5LJ3pEbr/9dr3++uuM94bJ6r32eDz69NNPtXfvXv+roqJC559/vvbu3asFCxbEq+mOEsnv9MKFC3X48GF9//33/mOfffaZUlNTNWvWrJi216kiuc99fX1KTQ18bKWlpUk6+S93TJ4tz8KYTY2dAnzLxrZu3Wra2trMqlWrTHZ2tvnf//5njDHm4YcfNrfeeqv/fN9yptWrV5u2tjazdetWlvaGwep9fv311016errZvHmz6ejo8L++++47u76CY1i918FYTRMeq/e5t7fXzJo1y9x4443mP//5j9m1a5c599xzzV133WXXV3AEq/f5lVdeMenp6aampsZ88cUXZvfu3aa4uNhceumldn0FR+jt7TWtra2mtbXVSDIbNmwwra2t/iXUU+FZmNBhxBhjNm/ebAoKCozL5TLz5883u3bt8v/dbbfdZq644oqA8//+97+bX/ziF8blcpmf/exnZsuWLXFusTNZuc9XXHGFkTTqddttt8W/4Q5k9Xd6JMJI+Kze53379pnFixebadOmmVmzZpnKykrT19cX51Y7j9X7vHHjRjN37lwzbdo0k5eXZ37961+bQ4cOxbnVzvK3v/1t3P/nToVnYYox9G0BAAD7JOycEQAA4AyEEQAAYCvCCAAAsBVhBAAA2IowAgAAbEUYAQAAtiKMAAAAWxFGAACArQgjAADAVoQRAABgK8IIAACwFWEEAADY6v8Du/mx8NJydUMAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "model_train_val_eval(train_X,val_X,train_y,val_y,model_pipeline)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "c6310fe5-e22b-4108-90c4-eb58f06a9207",
   "metadata": {},
   "outputs": [],
   "source": [
    "params = [\n",
    "    {\n",
    "    'model': [LogisticRegression()],\n",
    "    'model__penalty':['l2',None],\n",
    "    'model__C':[0.5,3]\n",
    "    }    \n",
    "]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "ed189572-bd45-4486-94ca-1485bef02e7b",
   "metadata": {},
   "outputs": [],
   "source": [
    "grid = GridSearchCV(estimator=model_pipeline, param_grid=params, \n",
    "                    cv=2, scoring='roc_auc')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "8543f4d9-4a5c-47ef-b171-29bcedf78c79",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-4 {color: black;background-color: white;}#sk-container-id-4 pre{padding: 0;}#sk-container-id-4 div.sk-toggleable {background-color: white;}#sk-container-id-4 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-4 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-4 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-4 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-4 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-4 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-4 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-4 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-4 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-4 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-4 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-4 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-4 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-4 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-4 div.sk-item {position: relative;z-index: 1;}#sk-container-id-4 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-4 div.sk-item::before, #sk-container-id-4 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-4 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-4 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-4 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-4 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-4 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-4 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-4 div.sk-label-container {text-align: center;}#sk-container-id-4 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-4 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-4\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>GridSearchCV(cv=2,\n",
       "             estimator=Pipeline(steps=[(&#x27;preprocess&#x27;,\n",
       "                                        ColumnTransformer(transformers=[(&#x27;cat_encode&#x27;,\n",
       "                                                                         Pipeline(steps=[(&#x27;impute_cat&#x27;,\n",
       "                                                                                          SimpleImputer(strategy=&#x27;most_frequent&#x27;)),\n",
       "                                                                                         (&#x27;ohe&#x27;,\n",
       "                                                                                          OneHotEncoder(handle_unknown=&#x27;ignore&#x27;))]),\n",
       "                                                                         [&#x27;department&#x27;,\n",
       "                                                                          &#x27;region&#x27;,\n",
       "                                                                          &#x27;education&#x27;,\n",
       "                                                                          &#x27;gender&#x27;,\n",
       "                                                                          &#x27;recruitment_channel&#x27;]),\n",
       "                                                                        (&#x27;num_encode&#x27;,\n",
       "                                                                         Pipeline(steps=[(&#x27;impute_num&#x27;,\n",
       "                                                                                          SimpleImputer(strategy=&#x27;median&#x27;)),\n",
       "                                                                                         (&#x27;scale&#x27;,\n",
       "                                                                                          StandardScaler())]),\n",
       "                                                                         [&#x27;no_of_trainings&#x27;,\n",
       "                                                                          &#x27;age&#x27;,\n",
       "                                                                          &#x27;previous_year_rating&#x27;,\n",
       "                                                                          &#x27;length_of_service&#x27;,\n",
       "                                                                          &#x27;KPIs_met &#x27;\n",
       "                                                                          &#x27;&gt;80%&#x27;,\n",
       "                                                                          &#x27;awards_won?&#x27;,\n",
       "                                                                          &#x27;avg_training_score&#x27;])])),\n",
       "                                       (&#x27;model&#x27;, LogisticRegression())]),\n",
       "             param_grid=[{&#x27;model&#x27;: [LogisticRegression(C=0.5)],\n",
       "                          &#x27;model__C&#x27;: [0.5, 3],\n",
       "                          &#x27;model__penalty&#x27;: [&#x27;l2&#x27;, None]}],\n",
       "             scoring=&#x27;roc_auc&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item sk-dashed-wrapped\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-29\" type=\"checkbox\" ><label for=\"sk-estimator-id-29\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">GridSearchCV</label><div class=\"sk-toggleable__content\"><pre>GridSearchCV(cv=2,\n",
       "             estimator=Pipeline(steps=[(&#x27;preprocess&#x27;,\n",
       "                                        ColumnTransformer(transformers=[(&#x27;cat_encode&#x27;,\n",
       "                                                                         Pipeline(steps=[(&#x27;impute_cat&#x27;,\n",
       "                                                                                          SimpleImputer(strategy=&#x27;most_frequent&#x27;)),\n",
       "                                                                                         (&#x27;ohe&#x27;,\n",
       "                                                                                          OneHotEncoder(handle_unknown=&#x27;ignore&#x27;))]),\n",
       "                                                                         [&#x27;department&#x27;,\n",
       "                                                                          &#x27;region&#x27;,\n",
       "                                                                          &#x27;education&#x27;,\n",
       "                                                                          &#x27;gender&#x27;,\n",
       "                                                                          &#x27;recruitment_channel&#x27;]),\n",
       "                                                                        (&#x27;num_encode&#x27;,\n",
       "                                                                         Pipeline(steps=[(&#x27;impute_num&#x27;,\n",
       "                                                                                          SimpleImputer(strategy=&#x27;median&#x27;)),\n",
       "                                                                                         (&#x27;scale&#x27;,\n",
       "                                                                                          StandardScaler())]),\n",
       "                                                                         [&#x27;no_of_trainings&#x27;,\n",
       "                                                                          &#x27;age&#x27;,\n",
       "                                                                          &#x27;previous_year_rating&#x27;,\n",
       "                                                                          &#x27;length_of_service&#x27;,\n",
       "                                                                          &#x27;KPIs_met &#x27;\n",
       "                                                                          &#x27;&gt;80%&#x27;,\n",
       "                                                                          &#x27;awards_won?&#x27;,\n",
       "                                                                          &#x27;avg_training_score&#x27;])])),\n",
       "                                       (&#x27;model&#x27;, LogisticRegression())]),\n",
       "             param_grid=[{&#x27;model&#x27;: [LogisticRegression(C=0.5)],\n",
       "                          &#x27;model__C&#x27;: [0.5, 3],\n",
       "                          &#x27;model__penalty&#x27;: [&#x27;l2&#x27;, None]}],\n",
       "             scoring=&#x27;roc_auc&#x27;)</pre></div></div></div><div class=\"sk-parallel\"><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-30\" type=\"checkbox\" ><label for=\"sk-estimator-id-30\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">estimator: Pipeline</label><div class=\"sk-toggleable__content\"><pre>Pipeline(steps=[(&#x27;preprocess&#x27;,\n",
       "                 ColumnTransformer(transformers=[(&#x27;cat_encode&#x27;,\n",
       "                                                  Pipeline(steps=[(&#x27;impute_cat&#x27;,\n",
       "                                                                   SimpleImputer(strategy=&#x27;most_frequent&#x27;)),\n",
       "                                                                  (&#x27;ohe&#x27;,\n",
       "                                                                   OneHotEncoder(handle_unknown=&#x27;ignore&#x27;))]),\n",
       "                                                  [&#x27;department&#x27;, &#x27;region&#x27;,\n",
       "                                                   &#x27;education&#x27;, &#x27;gender&#x27;,\n",
       "                                                   &#x27;recruitment_channel&#x27;]),\n",
       "                                                 (&#x27;num_encode&#x27;,\n",
       "                                                  Pipeline(steps=[(&#x27;impute_num&#x27;,\n",
       "                                                                   SimpleImputer(strategy=&#x27;median&#x27;)),\n",
       "                                                                  (&#x27;scale&#x27;,\n",
       "                                                                   StandardScaler())]),\n",
       "                                                  [&#x27;no_of_trainings&#x27;, &#x27;age&#x27;,\n",
       "                                                   &#x27;previous_year_rating&#x27;,\n",
       "                                                   &#x27;length_of_service&#x27;,\n",
       "                                                   &#x27;KPIs_met &gt;80%&#x27;,\n",
       "                                                   &#x27;awards_won?&#x27;,\n",
       "                                                   &#x27;avg_training_score&#x27;])])),\n",
       "                (&#x27;model&#x27;, LogisticRegression())])</pre></div></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-serial\"><div class=\"sk-item sk-dashed-wrapped\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-31\" type=\"checkbox\" ><label for=\"sk-estimator-id-31\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">preprocess: ColumnTransformer</label><div class=\"sk-toggleable__content\"><pre>ColumnTransformer(transformers=[(&#x27;cat_encode&#x27;,\n",
       "                                 Pipeline(steps=[(&#x27;impute_cat&#x27;,\n",
       "                                                  SimpleImputer(strategy=&#x27;most_frequent&#x27;)),\n",
       "                                                 (&#x27;ohe&#x27;,\n",
       "                                                  OneHotEncoder(handle_unknown=&#x27;ignore&#x27;))]),\n",
       "                                 [&#x27;department&#x27;, &#x27;region&#x27;, &#x27;education&#x27;, &#x27;gender&#x27;,\n",
       "                                  &#x27;recruitment_channel&#x27;]),\n",
       "                                (&#x27;num_encode&#x27;,\n",
       "                                 Pipeline(steps=[(&#x27;impute_num&#x27;,\n",
       "                                                  SimpleImputer(strategy=&#x27;median&#x27;)),\n",
       "                                                 (&#x27;scale&#x27;, StandardScaler())]),\n",
       "                                 [&#x27;no_of_trainings&#x27;, &#x27;age&#x27;,\n",
       "                                  &#x27;previous_year_rating&#x27;, &#x27;length_of_service&#x27;,\n",
       "                                  &#x27;KPIs_met &gt;80%&#x27;, &#x27;awards_won?&#x27;,\n",
       "                                  &#x27;avg_training_score&#x27;])])</pre></div></div></div><div class=\"sk-parallel\"><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-32\" type=\"checkbox\" ><label for=\"sk-estimator-id-32\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">cat_encode</label><div class=\"sk-toggleable__content\"><pre>[&#x27;department&#x27;, &#x27;region&#x27;, &#x27;education&#x27;, &#x27;gender&#x27;, &#x27;recruitment_channel&#x27;]</pre></div></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-33\" type=\"checkbox\" ><label for=\"sk-estimator-id-33\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">SimpleImputer</label><div class=\"sk-toggleable__content\"><pre>SimpleImputer(strategy=&#x27;most_frequent&#x27;)</pre></div></div></div><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-34\" type=\"checkbox\" ><label for=\"sk-estimator-id-34\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">OneHotEncoder</label><div class=\"sk-toggleable__content\"><pre>OneHotEncoder(handle_unknown=&#x27;ignore&#x27;)</pre></div></div></div></div></div></div></div></div><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-35\" type=\"checkbox\" ><label for=\"sk-estimator-id-35\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">num_encode</label><div class=\"sk-toggleable__content\"><pre>[&#x27;no_of_trainings&#x27;, &#x27;age&#x27;, &#x27;previous_year_rating&#x27;, &#x27;length_of_service&#x27;, &#x27;KPIs_met &gt;80%&#x27;, &#x27;awards_won?&#x27;, &#x27;avg_training_score&#x27;]</pre></div></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-36\" type=\"checkbox\" ><label for=\"sk-estimator-id-36\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">SimpleImputer</label><div class=\"sk-toggleable__content\"><pre>SimpleImputer(strategy=&#x27;median&#x27;)</pre></div></div></div><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-37\" type=\"checkbox\" ><label for=\"sk-estimator-id-37\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">StandardScaler</label><div class=\"sk-toggleable__content\"><pre>StandardScaler()</pre></div></div></div></div></div></div></div></div></div></div><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-38\" type=\"checkbox\" ><label for=\"sk-estimator-id-38\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">LogisticRegression</label><div class=\"sk-toggleable__content\"><pre>LogisticRegression()</pre></div></div></div></div></div></div></div></div></div></div></div></div>"
      ],
      "text/plain": [
       "GridSearchCV(cv=2,\n",
       "             estimator=Pipeline(steps=[('preprocess',\n",
       "                                        ColumnTransformer(transformers=[('cat_encode',\n",
       "                                                                         Pipeline(steps=[('impute_cat',\n",
       "                                                                                          SimpleImputer(strategy='most_frequent')),\n",
       "                                                                                         ('ohe',\n",
       "                                                                                          OneHotEncoder(handle_unknown='ignore'))]),\n",
       "                                                                         ['department',\n",
       "                                                                          'region',\n",
       "                                                                          'education',\n",
       "                                                                          'gender',\n",
       "                                                                          'recruitment_channel']),\n",
       "                                                                        ('num_encode',\n",
       "                                                                         Pipeline(steps=[('impute_num',\n",
       "                                                                                          SimpleImputer(strategy='median')),\n",
       "                                                                                         ('scale',\n",
       "                                                                                          StandardScaler())]),\n",
       "                                                                         ['no_of_trainings',\n",
       "                                                                          'age',\n",
       "                                                                          'previous_year_rating',\n",
       "                                                                          'length_of_service',\n",
       "                                                                          'KPIs_met '\n",
       "                                                                          '>80%',\n",
       "                                                                          'awards_won?',\n",
       "                                                                          'avg_training_score'])])),\n",
       "                                       ('model', LogisticRegression())]),\n",
       "             param_grid=[{'model': [LogisticRegression(C=0.5)],\n",
       "                          'model__C': [0.5, 3],\n",
       "                          'model__penalty': ['l2', None]}],\n",
       "             scoring='roc_auc')"
      ]
     },
     "execution_count": 61,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "grid.fit(train_X, train_y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "id": "26ffdf11-742a-46a7-8fbf-7a711599afa4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'model': LogisticRegression(C=0.5), 'model__C': 0.5, 'model__penalty': 'l2'}"
      ]
     },
     "execution_count": 64,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "grid.best_params_\n",
    "'''A value below 0.5 indicates a poor model. A value of 0.5 indicates that the model is no better out classifying outcomes than random chance.'''"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "d9370102-a353-4910-94ac-ed98006669d4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>params</th>\n",
       "      <th>mean_test_score</th>\n",
       "      <th>rank_test_score</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>{'model': LogisticRegression(C=0.5), 'model__C': 0.5, 'model__penalty': 'l2'}</td>\n",
       "      <td>0.872957</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>{'model': LogisticRegression(C=0.5), 'model__C': 0.5, 'model__penalty': None}</td>\n",
       "      <td>0.869788</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>{'model': LogisticRegression(C=0.5), 'model__C': 3, 'model__penalty': 'l2'}</td>\n",
       "      <td>0.870486</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>{'model': LogisticRegression(C=0.5), 'model__C': 3, 'model__penalty': None}</td>\n",
       "      <td>0.869788</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                                          params  \\\n",
       "0  {'model': LogisticRegression(C=0.5), 'model__C': 0.5, 'model__penalty': 'l2'}   \n",
       "1  {'model': LogisticRegression(C=0.5), 'model__C': 0.5, 'model__penalty': None}   \n",
       "2    {'model': LogisticRegression(C=0.5), 'model__C': 3, 'model__penalty': 'l2'}   \n",
       "3    {'model': LogisticRegression(C=0.5), 'model__C': 3, 'model__penalty': None}   \n",
       "\n",
       "   mean_test_score  rank_test_score  \n",
       "0         0.872957                1  \n",
       "1         0.869788                3  \n",
       "2         0.870486                2  \n",
       "3         0.869788                3  "
      ]
     },
     "execution_count": 63,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "res_df = pd.DataFrame(grid.cv_results_,)\n",
    "pd.set_option('display.max_colwidth',100)\n",
    "res_df[['params','mean_test_score','rank_test_score']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "id": "a499d868-1a0f-44a3-bfb9-92d47802b093",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>employee_id</th>\n",
       "      <th>is_promoted</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>8724</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>74430</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>72255</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   employee_id  is_promoted\n",
       "0         8724            0\n",
       "1        74430            0\n",
       "2        72255            0"
      ]
     },
     "execution_count": 65,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# read the submission file\n",
    "# predict with the last model\n",
    "#upload into the Analytic Vidya website\n",
    "\n",
    "sub = pd.read_csv('../Data/sample_submission_M0L0uXE.csv')\n",
    "sub.head(3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "id": "16e0c5a0-5c93-45d2-b566-2535da75fc44",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>employee_id</th>\n",
       "      <th>department</th>\n",
       "      <th>region</th>\n",
       "      <th>education</th>\n",
       "      <th>gender</th>\n",
       "      <th>recruitment_channel</th>\n",
       "      <th>no_of_trainings</th>\n",
       "      <th>age</th>\n",
       "      <th>previous_year_rating</th>\n",
       "      <th>length_of_service</th>\n",
       "      <th>KPIs_met &gt;80%</th>\n",
       "      <th>awards_won?</th>\n",
       "      <th>avg_training_score</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>8724</td>\n",
       "      <td>Technology</td>\n",
       "      <td>region_26</td>\n",
       "      <td>Bachelor's</td>\n",
       "      <td>m</td>\n",
       "      <td>sourcing</td>\n",
       "      <td>1</td>\n",
       "      <td>24</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>77</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>74430</td>\n",
       "      <td>HR</td>\n",
       "      <td>region_4</td>\n",
       "      <td>Bachelor's</td>\n",
       "      <td>f</td>\n",
       "      <td>other</td>\n",
       "      <td>1</td>\n",
       "      <td>31</td>\n",
       "      <td>3.0</td>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>51</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>72255</td>\n",
       "      <td>Sales &amp; Marketing</td>\n",
       "      <td>region_13</td>\n",
       "      <td>Bachelor's</td>\n",
       "      <td>m</td>\n",
       "      <td>other</td>\n",
       "      <td>1</td>\n",
       "      <td>31</td>\n",
       "      <td>1.0</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>47</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   employee_id         department     region   education gender  \\\n",
       "0         8724         Technology  region_26  Bachelor's      m   \n",
       "1        74430                 HR   region_4  Bachelor's      f   \n",
       "2        72255  Sales & Marketing  region_13  Bachelor's      m   \n",
       "\n",
       "  recruitment_channel  no_of_trainings  age  previous_year_rating  \\\n",
       "0            sourcing                1   24                   NaN   \n",
       "1               other                1   31                   3.0   \n",
       "2               other                1   31                   1.0   \n",
       "\n",
       "   length_of_service  KPIs_met >80%  awards_won?  avg_training_score  \n",
       "0                  1              1            0                  77  \n",
       "1                  5              0            0                  51  \n",
       "2                  4              0            0                  47  "
      ]
     },
     "execution_count": 66,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test.head(3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "1c272176-4c17-44a8-8df7-bc05d462769f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['is_promoted'], dtype='object')"
      ]
     },
     "execution_count": 67,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.columns.difference(test.columns)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "id": "26cfdfb4-e51b-476f-83a3-14e25562d6ed",
   "metadata": {},
   "outputs": [],
   "source": [
    "# updating the existing promoted values with predicted values\n",
    "sub['is_promoted'] = model_pipeline.predict(test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "id": "1272f449-6e58-4d8a-a39a-2d137657e0ea",
   "metadata": {},
   "outputs": [],
   "source": [
    "sub.to_csv('sub_1.csv',index=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "id": "ef808f65-a0db-46bc-8fdc-749e8736ede0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>employee_id</th>\n",
       "      <th>is_promoted</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>8724</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>74430</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>72255</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>38562</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>64486</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23485</th>\n",
       "      <td>53478</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23486</th>\n",
       "      <td>25600</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23487</th>\n",
       "      <td>45409</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23488</th>\n",
       "      <td>1186</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23489</th>\n",
       "      <td>5973</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>23490 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "       employee_id  is_promoted\n",
       "0             8724            0\n",
       "1            74430            0\n",
       "2            72255            0\n",
       "3            38562            0\n",
       "4            64486            0\n",
       "...            ...          ...\n",
       "23485        53478            0\n",
       "23486        25600            0\n",
       "23487        45409            0\n",
       "23488         1186            0\n",
       "23489         5973            1\n",
       "\n",
       "[23490 rows x 2 columns]"
      ]
     },
     "execution_count": 70,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sub"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "id": "4330971d-3886-4e41-a98e-b25a6074dd52",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['jobchg_pipeline_model.pkl']"
      ]
     },
     "execution_count": 73,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "joblib.dump(model_pipeline,'jobchg_pipeline_model.pkl')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "id": "08528151-07b2-45f1-86fd-ccb9dfcb7acb",
   "metadata": {},
   "outputs": [],
   "source": [
    "from imblearn.over_sampling import RandomOverSampler"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "id": "7ab540fc-e94a-4a96-96f6-f2504365bc3d",
   "metadata": {},
   "outputs": [],
   "source": [
    "over_sampling = RandomOverSampler()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "id": "fc5b35ec-7ee2-46df-8390-f822c1fcefb6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "is_promoted\n",
       "0              45090\n",
       "1               4237\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 76,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_y.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "id": "1a054162-97c4-4f0d-80db-18382024564e",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_X_os, train_y_os = over_sampling.fit_resample(train_X,train_y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "id": "2d707da1-9970-4e2f-aadc-309616469e5a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "is_promoted\n",
       "0              45090\n",
       "1              45090\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 78,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_y_os.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "id": "1ed7212c-5999-415a-af93-855aa7b7a008",
   "metadata": {},
   "outputs": [],
   "source": [
    "params_2 = [\n",
    "    {\n",
    "    'model': [LogisticRegression()],\n",
    "    'model__penalty':['l2',None],\n",
    "    'model__C':[0.5,3]\n",
    "    },\n",
    "    {\n",
    "    'model': [DecisionTreeClassifier()],\n",
    "    'model__max_depth':[3,5]\n",
    "    }\n",
    "]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "id": "42acc1ad-204c-4a27-b2b9-2ac23912ca54",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[{'model': [LogisticRegression()],\n",
       "  'model__penalty': ['l2', None],\n",
       "  'model__C': [0.5, 3]},\n",
       " {'model': [DecisionTreeClassifier()], 'model__max_depth': [3, 5]}]"
      ]
     },
     "execution_count": 80,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "params_2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "id": "3b8c2343-18d0-4db2-b546-7c5da28a882d",
   "metadata": {},
   "outputs": [],
   "source": [
    "grid_2 = GridSearchCV(estimator=model_pipeline, param_grid=params_2, \n",
    "                    cv=2, scoring='roc_auc')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "id": "5430d446-7163-4fd1-8b23-958b696b929c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-5 {color: black;background-color: white;}#sk-container-id-5 pre{padding: 0;}#sk-container-id-5 div.sk-toggleable {background-color: white;}#sk-container-id-5 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-5 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-5 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-5 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-5 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-5 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-5 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-5 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-5 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-5 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-5 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-5 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-5 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-5 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-5 div.sk-item {position: relative;z-index: 1;}#sk-container-id-5 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-5 div.sk-item::before, #sk-container-id-5 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-5 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-5 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-5 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-5 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-5 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-5 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-5 div.sk-label-container {text-align: center;}#sk-container-id-5 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-5 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-5\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>GridSearchCV(cv=2,\n",
       "             estimator=Pipeline(steps=[(&#x27;preprocess&#x27;,\n",
       "                                        ColumnTransformer(transformers=[(&#x27;cat_encode&#x27;,\n",
       "                                                                         Pipeline(steps=[(&#x27;impute_cat&#x27;,\n",
       "                                                                                          SimpleImputer(strategy=&#x27;most_frequent&#x27;)),\n",
       "                                                                                         (&#x27;ohe&#x27;,\n",
       "                                                                                          OneHotEncoder(handle_unknown=&#x27;ignore&#x27;))]),\n",
       "                                                                         [&#x27;department&#x27;,\n",
       "                                                                          &#x27;region&#x27;,\n",
       "                                                                          &#x27;education&#x27;,\n",
       "                                                                          &#x27;gender&#x27;,\n",
       "                                                                          &#x27;recruitment_channel&#x27;]),\n",
       "                                                                        (&#x27;num_encode&#x27;,\n",
       "                                                                         Pipeline(steps=[(&#x27;impute_num&#x27;,\n",
       "                                                                                          SimpleImputer(...\n",
       "                                                                         [&#x27;no_of_trainings&#x27;,\n",
       "                                                                          &#x27;age&#x27;,\n",
       "                                                                          &#x27;previous_year_rating&#x27;,\n",
       "                                                                          &#x27;length_of_service&#x27;,\n",
       "                                                                          &#x27;KPIs_met &#x27;\n",
       "                                                                          &#x27;&gt;80%&#x27;,\n",
       "                                                                          &#x27;awards_won?&#x27;,\n",
       "                                                                          &#x27;avg_training_score&#x27;])])),\n",
       "                                       (&#x27;model&#x27;, LogisticRegression())]),\n",
       "             param_grid=[{&#x27;model&#x27;: [LogisticRegression(C=0.5, penalty=None)],\n",
       "                          &#x27;model__C&#x27;: [0.5, 3],\n",
       "                          &#x27;model__penalty&#x27;: [&#x27;l2&#x27;, None]},\n",
       "                         {&#x27;model&#x27;: [DecisionTreeClassifier()],\n",
       "                          &#x27;model__max_depth&#x27;: [3, 5]}],\n",
       "             scoring=&#x27;roc_auc&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item sk-dashed-wrapped\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-39\" type=\"checkbox\" ><label for=\"sk-estimator-id-39\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">GridSearchCV</label><div class=\"sk-toggleable__content\"><pre>GridSearchCV(cv=2,\n",
       "             estimator=Pipeline(steps=[(&#x27;preprocess&#x27;,\n",
       "                                        ColumnTransformer(transformers=[(&#x27;cat_encode&#x27;,\n",
       "                                                                         Pipeline(steps=[(&#x27;impute_cat&#x27;,\n",
       "                                                                                          SimpleImputer(strategy=&#x27;most_frequent&#x27;)),\n",
       "                                                                                         (&#x27;ohe&#x27;,\n",
       "                                                                                          OneHotEncoder(handle_unknown=&#x27;ignore&#x27;))]),\n",
       "                                                                         [&#x27;department&#x27;,\n",
       "                                                                          &#x27;region&#x27;,\n",
       "                                                                          &#x27;education&#x27;,\n",
       "                                                                          &#x27;gender&#x27;,\n",
       "                                                                          &#x27;recruitment_channel&#x27;]),\n",
       "                                                                        (&#x27;num_encode&#x27;,\n",
       "                                                                         Pipeline(steps=[(&#x27;impute_num&#x27;,\n",
       "                                                                                          SimpleImputer(...\n",
       "                                                                         [&#x27;no_of_trainings&#x27;,\n",
       "                                                                          &#x27;age&#x27;,\n",
       "                                                                          &#x27;previous_year_rating&#x27;,\n",
       "                                                                          &#x27;length_of_service&#x27;,\n",
       "                                                                          &#x27;KPIs_met &#x27;\n",
       "                                                                          &#x27;&gt;80%&#x27;,\n",
       "                                                                          &#x27;awards_won?&#x27;,\n",
       "                                                                          &#x27;avg_training_score&#x27;])])),\n",
       "                                       (&#x27;model&#x27;, LogisticRegression())]),\n",
       "             param_grid=[{&#x27;model&#x27;: [LogisticRegression(C=0.5, penalty=None)],\n",
       "                          &#x27;model__C&#x27;: [0.5, 3],\n",
       "                          &#x27;model__penalty&#x27;: [&#x27;l2&#x27;, None]},\n",
       "                         {&#x27;model&#x27;: [DecisionTreeClassifier()],\n",
       "                          &#x27;model__max_depth&#x27;: [3, 5]}],\n",
       "             scoring=&#x27;roc_auc&#x27;)</pre></div></div></div><div class=\"sk-parallel\"><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-40\" type=\"checkbox\" ><label for=\"sk-estimator-id-40\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">estimator: Pipeline</label><div class=\"sk-toggleable__content\"><pre>Pipeline(steps=[(&#x27;preprocess&#x27;,\n",
       "                 ColumnTransformer(transformers=[(&#x27;cat_encode&#x27;,\n",
       "                                                  Pipeline(steps=[(&#x27;impute_cat&#x27;,\n",
       "                                                                   SimpleImputer(strategy=&#x27;most_frequent&#x27;)),\n",
       "                                                                  (&#x27;ohe&#x27;,\n",
       "                                                                   OneHotEncoder(handle_unknown=&#x27;ignore&#x27;))]),\n",
       "                                                  [&#x27;department&#x27;, &#x27;region&#x27;,\n",
       "                                                   &#x27;education&#x27;, &#x27;gender&#x27;,\n",
       "                                                   &#x27;recruitment_channel&#x27;]),\n",
       "                                                 (&#x27;num_encode&#x27;,\n",
       "                                                  Pipeline(steps=[(&#x27;impute_num&#x27;,\n",
       "                                                                   SimpleImputer(strategy=&#x27;median&#x27;)),\n",
       "                                                                  (&#x27;scale&#x27;,\n",
       "                                                                   StandardScaler())]),\n",
       "                                                  [&#x27;no_of_trainings&#x27;, &#x27;age&#x27;,\n",
       "                                                   &#x27;previous_year_rating&#x27;,\n",
       "                                                   &#x27;length_of_service&#x27;,\n",
       "                                                   &#x27;KPIs_met &gt;80%&#x27;,\n",
       "                                                   &#x27;awards_won?&#x27;,\n",
       "                                                   &#x27;avg_training_score&#x27;])])),\n",
       "                (&#x27;model&#x27;, LogisticRegression())])</pre></div></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-serial\"><div class=\"sk-item sk-dashed-wrapped\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-41\" type=\"checkbox\" ><label for=\"sk-estimator-id-41\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">preprocess: ColumnTransformer</label><div class=\"sk-toggleable__content\"><pre>ColumnTransformer(transformers=[(&#x27;cat_encode&#x27;,\n",
       "                                 Pipeline(steps=[(&#x27;impute_cat&#x27;,\n",
       "                                                  SimpleImputer(strategy=&#x27;most_frequent&#x27;)),\n",
       "                                                 (&#x27;ohe&#x27;,\n",
       "                                                  OneHotEncoder(handle_unknown=&#x27;ignore&#x27;))]),\n",
       "                                 [&#x27;department&#x27;, &#x27;region&#x27;, &#x27;education&#x27;, &#x27;gender&#x27;,\n",
       "                                  &#x27;recruitment_channel&#x27;]),\n",
       "                                (&#x27;num_encode&#x27;,\n",
       "                                 Pipeline(steps=[(&#x27;impute_num&#x27;,\n",
       "                                                  SimpleImputer(strategy=&#x27;median&#x27;)),\n",
       "                                                 (&#x27;scale&#x27;, StandardScaler())]),\n",
       "                                 [&#x27;no_of_trainings&#x27;, &#x27;age&#x27;,\n",
       "                                  &#x27;previous_year_rating&#x27;, &#x27;length_of_service&#x27;,\n",
       "                                  &#x27;KPIs_met &gt;80%&#x27;, &#x27;awards_won?&#x27;,\n",
       "                                  &#x27;avg_training_score&#x27;])])</pre></div></div></div><div class=\"sk-parallel\"><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-42\" type=\"checkbox\" ><label for=\"sk-estimator-id-42\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">cat_encode</label><div class=\"sk-toggleable__content\"><pre>[&#x27;department&#x27;, &#x27;region&#x27;, &#x27;education&#x27;, &#x27;gender&#x27;, &#x27;recruitment_channel&#x27;]</pre></div></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-43\" type=\"checkbox\" ><label for=\"sk-estimator-id-43\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">SimpleImputer</label><div class=\"sk-toggleable__content\"><pre>SimpleImputer(strategy=&#x27;most_frequent&#x27;)</pre></div></div></div><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-44\" type=\"checkbox\" ><label for=\"sk-estimator-id-44\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">OneHotEncoder</label><div class=\"sk-toggleable__content\"><pre>OneHotEncoder(handle_unknown=&#x27;ignore&#x27;)</pre></div></div></div></div></div></div></div></div><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-45\" type=\"checkbox\" ><label for=\"sk-estimator-id-45\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">num_encode</label><div class=\"sk-toggleable__content\"><pre>[&#x27;no_of_trainings&#x27;, &#x27;age&#x27;, &#x27;previous_year_rating&#x27;, &#x27;length_of_service&#x27;, &#x27;KPIs_met &gt;80%&#x27;, &#x27;awards_won?&#x27;, &#x27;avg_training_score&#x27;]</pre></div></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-46\" type=\"checkbox\" ><label for=\"sk-estimator-id-46\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">SimpleImputer</label><div class=\"sk-toggleable__content\"><pre>SimpleImputer(strategy=&#x27;median&#x27;)</pre></div></div></div><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-47\" type=\"checkbox\" ><label for=\"sk-estimator-id-47\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">StandardScaler</label><div class=\"sk-toggleable__content\"><pre>StandardScaler()</pre></div></div></div></div></div></div></div></div></div></div><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-48\" type=\"checkbox\" ><label for=\"sk-estimator-id-48\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">LogisticRegression</label><div class=\"sk-toggleable__content\"><pre>LogisticRegression()</pre></div></div></div></div></div></div></div></div></div></div></div></div>"
      ],
      "text/plain": [
       "GridSearchCV(cv=2,\n",
       "             estimator=Pipeline(steps=[('preprocess',\n",
       "                                        ColumnTransformer(transformers=[('cat_encode',\n",
       "                                                                         Pipeline(steps=[('impute_cat',\n",
       "                                                                                          SimpleImputer(strategy='most_frequent')),\n",
       "                                                                                         ('ohe',\n",
       "                                                                                          OneHotEncoder(handle_unknown='ignore'))]),\n",
       "                                                                         ['department',\n",
       "                                                                          'region',\n",
       "                                                                          'education',\n",
       "                                                                          'gender',\n",
       "                                                                          'recruitment_channel']),\n",
       "                                                                        ('num_encode',\n",
       "                                                                         Pipeline(steps=[('impute_num',\n",
       "                                                                                          SimpleImputer(...\n",
       "                                                                         ['no_of_trainings',\n",
       "                                                                          'age',\n",
       "                                                                          'previous_year_rating',\n",
       "                                                                          'length_of_service',\n",
       "                                                                          'KPIs_met '\n",
       "                                                                          '>80%',\n",
       "                                                                          'awards_won?',\n",
       "                                                                          'avg_training_score'])])),\n",
       "                                       ('model', LogisticRegression())]),\n",
       "             param_grid=[{'model': [LogisticRegression(C=0.5, penalty=None)],\n",
       "                          'model__C': [0.5, 3],\n",
       "                          'model__penalty': ['l2', None]},\n",
       "                         {'model': [DecisionTreeClassifier()],\n",
       "                          'model__max_depth': [3, 5]}],\n",
       "             scoring='roc_auc')"
      ]
     },
     "execution_count": 82,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "grid_2.fit(train_X_os, train_y_os)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "id": "a1b2733f-20f4-4d25-a2be-ec3811eb9471",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'model': LogisticRegression(C=0.5, penalty=None),\n",
       " 'model__C': 0.5,\n",
       " 'model__penalty': None}"
      ]
     },
     "execution_count": 84,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "grid_2.best_params_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "id": "51200e8d-68e9-4cd7-bea6-83c2217c08b8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-7 {color: black;background-color: white;}#sk-container-id-7 pre{padding: 0;}#sk-container-id-7 div.sk-toggleable {background-color: white;}#sk-container-id-7 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-7 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-7 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-7 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-7 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-7 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-7 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-7 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-7 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-7 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-7 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-7 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-7 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-7 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-7 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-7 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-7 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-7 div.sk-item {position: relative;z-index: 1;}#sk-container-id-7 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-7 div.sk-item::before, #sk-container-id-7 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-7 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-7 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-7 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-7 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-7 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-7 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-7 div.sk-label-container {text-align: center;}#sk-container-id-7 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-7 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-7\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>Pipeline(steps=[(&#x27;preprocess&#x27;,\n",
       "                 ColumnTransformer(transformers=[(&#x27;cat_encode&#x27;,\n",
       "                                                  Pipeline(steps=[(&#x27;impute_cat&#x27;,\n",
       "                                                                   SimpleImputer(strategy=&#x27;most_frequent&#x27;)),\n",
       "                                                                  (&#x27;ohe&#x27;,\n",
       "                                                                   OneHotEncoder(handle_unknown=&#x27;ignore&#x27;))]),\n",
       "                                                  [&#x27;department&#x27;, &#x27;region&#x27;,\n",
       "                                                   &#x27;education&#x27;, &#x27;gender&#x27;,\n",
       "                                                   &#x27;recruitment_channel&#x27;]),\n",
       "                                                 (&#x27;num_encode&#x27;,\n",
       "                                                  Pipeline(steps=[(&#x27;impute_num&#x27;,\n",
       "                                                                   SimpleImputer(strategy=&#x27;median&#x27;)),\n",
       "                                                                  (&#x27;scale&#x27;,\n",
       "                                                                   StandardScaler())]),\n",
       "                                                  [&#x27;no_of_trainings&#x27;, &#x27;age&#x27;,\n",
       "                                                   &#x27;previous_year_rating&#x27;,\n",
       "                                                   &#x27;length_of_service&#x27;,\n",
       "                                                   &#x27;KPIs_met &gt;80%&#x27;,\n",
       "                                                   &#x27;awards_won?&#x27;,\n",
       "                                                   &#x27;avg_training_score&#x27;])])),\n",
       "                (&#x27;model&#x27;, LogisticRegression(C=0.5, penalty=None))])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item sk-dashed-wrapped\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-59\" type=\"checkbox\" ><label for=\"sk-estimator-id-59\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">Pipeline</label><div class=\"sk-toggleable__content\"><pre>Pipeline(steps=[(&#x27;preprocess&#x27;,\n",
       "                 ColumnTransformer(transformers=[(&#x27;cat_encode&#x27;,\n",
       "                                                  Pipeline(steps=[(&#x27;impute_cat&#x27;,\n",
       "                                                                   SimpleImputer(strategy=&#x27;most_frequent&#x27;)),\n",
       "                                                                  (&#x27;ohe&#x27;,\n",
       "                                                                   OneHotEncoder(handle_unknown=&#x27;ignore&#x27;))]),\n",
       "                                                  [&#x27;department&#x27;, &#x27;region&#x27;,\n",
       "                                                   &#x27;education&#x27;, &#x27;gender&#x27;,\n",
       "                                                   &#x27;recruitment_channel&#x27;]),\n",
       "                                                 (&#x27;num_encode&#x27;,\n",
       "                                                  Pipeline(steps=[(&#x27;impute_num&#x27;,\n",
       "                                                                   SimpleImputer(strategy=&#x27;median&#x27;)),\n",
       "                                                                  (&#x27;scale&#x27;,\n",
       "                                                                   StandardScaler())]),\n",
       "                                                  [&#x27;no_of_trainings&#x27;, &#x27;age&#x27;,\n",
       "                                                   &#x27;previous_year_rating&#x27;,\n",
       "                                                   &#x27;length_of_service&#x27;,\n",
       "                                                   &#x27;KPIs_met &gt;80%&#x27;,\n",
       "                                                   &#x27;awards_won?&#x27;,\n",
       "                                                   &#x27;avg_training_score&#x27;])])),\n",
       "                (&#x27;model&#x27;, LogisticRegression(C=0.5, penalty=None))])</pre></div></div></div><div class=\"sk-serial\"><div class=\"sk-item sk-dashed-wrapped\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-60\" type=\"checkbox\" ><label for=\"sk-estimator-id-60\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">preprocess: ColumnTransformer</label><div class=\"sk-toggleable__content\"><pre>ColumnTransformer(transformers=[(&#x27;cat_encode&#x27;,\n",
       "                                 Pipeline(steps=[(&#x27;impute_cat&#x27;,\n",
       "                                                  SimpleImputer(strategy=&#x27;most_frequent&#x27;)),\n",
       "                                                 (&#x27;ohe&#x27;,\n",
       "                                                  OneHotEncoder(handle_unknown=&#x27;ignore&#x27;))]),\n",
       "                                 [&#x27;department&#x27;, &#x27;region&#x27;, &#x27;education&#x27;, &#x27;gender&#x27;,\n",
       "                                  &#x27;recruitment_channel&#x27;]),\n",
       "                                (&#x27;num_encode&#x27;,\n",
       "                                 Pipeline(steps=[(&#x27;impute_num&#x27;,\n",
       "                                                  SimpleImputer(strategy=&#x27;median&#x27;)),\n",
       "                                                 (&#x27;scale&#x27;, StandardScaler())]),\n",
       "                                 [&#x27;no_of_trainings&#x27;, &#x27;age&#x27;,\n",
       "                                  &#x27;previous_year_rating&#x27;, &#x27;length_of_service&#x27;,\n",
       "                                  &#x27;KPIs_met &gt;80%&#x27;, &#x27;awards_won?&#x27;,\n",
       "                                  &#x27;avg_training_score&#x27;])])</pre></div></div></div><div class=\"sk-parallel\"><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-61\" type=\"checkbox\" ><label for=\"sk-estimator-id-61\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">cat_encode</label><div class=\"sk-toggleable__content\"><pre>[&#x27;department&#x27;, &#x27;region&#x27;, &#x27;education&#x27;, &#x27;gender&#x27;, &#x27;recruitment_channel&#x27;]</pre></div></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-62\" type=\"checkbox\" ><label for=\"sk-estimator-id-62\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">SimpleImputer</label><div class=\"sk-toggleable__content\"><pre>SimpleImputer(strategy=&#x27;most_frequent&#x27;)</pre></div></div></div><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-63\" type=\"checkbox\" ><label for=\"sk-estimator-id-63\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">OneHotEncoder</label><div class=\"sk-toggleable__content\"><pre>OneHotEncoder(handle_unknown=&#x27;ignore&#x27;)</pre></div></div></div></div></div></div></div></div><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-64\" type=\"checkbox\" ><label for=\"sk-estimator-id-64\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">num_encode</label><div class=\"sk-toggleable__content\"><pre>[&#x27;no_of_trainings&#x27;, &#x27;age&#x27;, &#x27;previous_year_rating&#x27;, &#x27;length_of_service&#x27;, &#x27;KPIs_met &gt;80%&#x27;, &#x27;awards_won?&#x27;, &#x27;avg_training_score&#x27;]</pre></div></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-65\" type=\"checkbox\" ><label for=\"sk-estimator-id-65\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">SimpleImputer</label><div class=\"sk-toggleable__content\"><pre>SimpleImputer(strategy=&#x27;median&#x27;)</pre></div></div></div><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-66\" type=\"checkbox\" ><label for=\"sk-estimator-id-66\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">StandardScaler</label><div class=\"sk-toggleable__content\"><pre>StandardScaler()</pre></div></div></div></div></div></div></div></div></div></div><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-67\" type=\"checkbox\" ><label for=\"sk-estimator-id-67\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">LogisticRegression</label><div class=\"sk-toggleable__content\"><pre>LogisticRegression(C=0.5, penalty=None)</pre></div></div></div></div></div></div></div>"
      ],
      "text/plain": [
       "Pipeline(steps=[('preprocess',\n",
       "                 ColumnTransformer(transformers=[('cat_encode',\n",
       "                                                  Pipeline(steps=[('impute_cat',\n",
       "                                                                   SimpleImputer(strategy='most_frequent')),\n",
       "                                                                  ('ohe',\n",
       "                                                                   OneHotEncoder(handle_unknown='ignore'))]),\n",
       "                                                  ['department', 'region',\n",
       "                                                   'education', 'gender',\n",
       "                                                   'recruitment_channel']),\n",
       "                                                 ('num_encode',\n",
       "                                                  Pipeline(steps=[('impute_num',\n",
       "                                                                   SimpleImputer(strategy='median')),\n",
       "                                                                  ('scale',\n",
       "                                                                   StandardScaler())]),\n",
       "                                                  ['no_of_trainings', 'age',\n",
       "                                                   'previous_year_rating',\n",
       "                                                   'length_of_service',\n",
       "                                                   'KPIs_met >80%',\n",
       "                                                   'awards_won?',\n",
       "                                                   'avg_training_score'])])),\n",
       "                ('model', LogisticRegression(C=0.5, penalty=None))])"
      ]
     },
     "execution_count": 85,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "grid_2.best_estimator_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "id": "93515527-92a8-4b11-9bf2-70b28d3d73e9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'mean_fit_time': array([0.43982661, 0.4467473 , 0.46208274, 0.46654105, 0.22792482,\n",
       "        0.26952755]),\n",
       " 'std_fit_time': array([0.03173363, 0.01268935, 0.03042901, 0.01497698, 0.039011  ,\n",
       "        0.01411498]),\n",
       " 'mean_score_time': array([0.08284628, 0.08010364, 0.07887292, 0.11284482, 0.0844754 ,\n",
       "        0.08931005]),\n",
       " 'std_score_time': array([4.23943996e-03, 1.30391121e-03, 5.69820404e-05, 1.25025511e-02,\n",
       "        5.82325459e-03, 5.84638119e-03]),\n",
       " 'param_model': masked_array(data=[LogisticRegression(C=0.5, penalty=None),\n",
       "                    LogisticRegression(C=0.5, penalty=None),\n",
       "                    LogisticRegression(C=0.5, penalty=None),\n",
       "                    LogisticRegression(C=0.5, penalty=None),\n",
       "                    DecisionTreeClassifier(), DecisionTreeClassifier()],\n",
       "              mask=[False, False, False, False, False, False],\n",
       "        fill_value='?',\n",
       "             dtype=object),\n",
       " 'param_model__C': masked_array(data=[0.5, 0.5, 3, 3, --, --],\n",
       "              mask=[False, False, False, False,  True,  True],\n",
       "        fill_value='?',\n",
       "             dtype=object),\n",
       " 'param_model__penalty': masked_array(data=['l2', None, 'l2', None, --, --],\n",
       "              mask=[False, False, False, False,  True,  True],\n",
       "        fill_value='?',\n",
       "             dtype=object),\n",
       " 'param_model__max_depth': masked_array(data=[--, --, --, --, 3, 5],\n",
       "              mask=[ True,  True,  True,  True, False, False],\n",
       "        fill_value='?',\n",
       "             dtype=object),\n",
       " 'params': [{'model': LogisticRegression(C=0.5, penalty=None),\n",
       "   'model__C': 0.5,\n",
       "   'model__penalty': 'l2'},\n",
       "  {'model': LogisticRegression(C=0.5, penalty=None),\n",
       "   'model__C': 0.5,\n",
       "   'model__penalty': None},\n",
       "  {'model': LogisticRegression(C=0.5, penalty=None),\n",
       "   'model__C': 3,\n",
       "   'model__penalty': 'l2'},\n",
       "  {'model': LogisticRegression(C=0.5, penalty=None),\n",
       "   'model__C': 3,\n",
       "   'model__penalty': None},\n",
       "  {'model': DecisionTreeClassifier(), 'model__max_depth': 3},\n",
       "  {'model': DecisionTreeClassifier(), 'model__max_depth': 5}],\n",
       " 'split0_test_score': array([0.87698409, 0.87723922, 0.87721022, 0.87723922, 0.80099508,\n",
       "        0.84970439]),\n",
       " 'split1_test_score': array([0.87514791, 0.87550394, 0.8754446 , 0.87550394, 0.79655006,\n",
       "        0.84713777]),\n",
       " 'mean_test_score': array([0.876066  , 0.87637158, 0.87632741, 0.87637158, 0.79877257,\n",
       "        0.84842108]),\n",
       " 'std_test_score': array([0.00091809, 0.00086764, 0.00088281, 0.00086764, 0.00222251,\n",
       "        0.00128331]),\n",
       " 'rank_test_score': array([4, 1, 3, 1, 6, 5])}"
      ]
     },
     "execution_count": 86,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "grid_2.cv_results_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "id": "4fa9310c-c251-4350-a72b-ba8d6eeb3779",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train AUC\n",
      "0.7920591905638805\n",
      "Valid AUC\n",
      "0.7971944131768165\n",
      "Train cnf_matrix\n",
      "[[34479 10611]\n",
      " [  765  3472]]\n",
      "Valid cnf_matrix\n",
      "[[3857 1193]\n",
      " [  73  358]]\n",
      "Train cls_rep\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.98      0.76      0.86     45090\n",
      "           1       0.25      0.82      0.38      4237\n",
      "\n",
      "    accuracy                           0.77     49327\n",
      "   macro avg       0.61      0.79      0.62     49327\n",
      "weighted avg       0.92      0.77      0.82     49327\n",
      "\n",
      "Valid cls rep\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.98      0.76      0.86      5050\n",
      "           1       0.23      0.83      0.36       431\n",
      "\n",
      "    accuracy                           0.77      5481\n",
      "   macro avg       0.61      0.80      0.61      5481\n",
      "weighted avg       0.92      0.77      0.82      5481\n",
      "\n"
     ]
    },
    {
     "data": {
      "image/png": "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cggAAIlFGAEAALaKKow0NDSotLRUubm5KisrU0tLy7jnvvXWW7riiit0yimnyOPxqKKiQu+//37UDQYAAKnFchhpbGzUypUrtXr1anV0dGjhwoVasmSJurq6wp7/wQcf6IorrlBTU5Pa29t16aWX6uqrr1ZHR8ekGw8AAJwvwxhr+9TOmzdPF110kTZs2OA/NmvWLF177bWqr6+P6Gecd955qq6u1sMPPxzR+V6vVwUFBerv75fH47HS3IgcGjqq89Yc663pXLdYea7smH8GAADpJtLnt6WekeHhYbW3t6uqqiroeFVVlXbt2hXRzxgdHdXAwIBOPvnkcc8ZGhqS1+sNesULO/UCAGAvS2Gkt7dXIyMjKiwsDDpeWFionp6eiH7GE088oUOHDunGG28c95z6+noVFBT4X8XFxVaaaQk1RgAAsFdUE1hD63AYYyKqzfH666/rkUceUWNjo0499dRxz6urq1N/f7//tW/fvmiaaRk1RgAASDxLkyOmTZumrKysMb0gBw8eHNNbEqqxsVHLli3T1q1btWjRognPdbvdcrvdVpoWtcAZM+QQAAASz1LPiMvlUllZmZqbm4OONzc3q7KyctzrXn/9dd1666167bXXdNVVV0XX0jhgvggAAPazvGyktrZWN998s8rLy1VRUaEXXnhBXV1dqqmpkXRsiOXLL7/Uq6++KulYEFm6dKmeeuopXXzxxf5elSlTpqigoCCGX8U65osAAGA/y2GkurpafX19Wrdunbq7uzVnzhw1NTWppKREktTd3R1Uc+T555/X0aNHdffdd+vuu+/2H7/lllv0yiuvTP4bxAjzRQAAsIflOiN2iFedkcHho5r9MPVFAACIh7jUGQEAAIg1wggAALAVYQQAANiKMAIAAGxFGAEAALYijAAAAFsRRgAAgK0IIwAAwFaEEQAAYCvCCAAAsFVah5HkL4QPAEDqS9swYozRDRtb7W4GAABpL23DyOEjI+rs9kqSZhd5NCUny+YWAQCQntI2jATaWlOhjIwMu5sBAEBaIoxIIocAAGAfwggAALAVYQQAANiKMAIAAGxFGAEAALYijAAAAFsRRgAAgK0IIwAAwFaEEQAAYCvCCAAAsBVhBAAA2IowAgAAbEUYAQAAtiKMAAAAWxFGAACArQgjAADAVoQRAABgK8IIAACwFWEEAADYijACAABsRRgB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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "new_model = grid_2.best_estimator_\n",
    "model_train_val_eval(train_X,val_X,train_y,val_y,new_model)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "id": "95df6218-a0ea-4e07-b9a1-61ce73ad03ef",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train AUC\n",
      "0.7914282546019072\n",
      "Valid AUC\n",
      "0.7971944131768165\n",
      "Train cnf_matrix\n",
      "[[34479 10611]\n",
      " [ 8198 36892]]\n",
      "Valid cnf_matrix\n",
      "[[3857 1193]\n",
      " [  73  358]]\n",
      "Train cls_rep\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.81      0.76      0.79     45090\n",
      "           1       0.78      0.82      0.80     45090\n",
      "\n",
      "    accuracy                           0.79     90180\n",
      "   macro avg       0.79      0.79      0.79     90180\n",
      "weighted avg       0.79      0.79      0.79     90180\n",
      "\n",
      "Valid cls rep\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.98      0.76      0.86      5050\n",
      "           1       0.23      0.83      0.36       431\n",
      "\n",
      "    accuracy                           0.77      5481\n",
      "   macro avg       0.61      0.80      0.61      5481\n",
      "weighted avg       0.92      0.77      0.82      5481\n",
      "\n"
     ]
    },
    {
     "data": {
      "image/png": "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cggAAIlFGAEAALaKKow0NDSotLRUubm5KisrU0tLy7jnvvXWW7riiit0yimnyOPxqKKiQu+//37UDQYAAKnFchhpbGzUypUrtXr1anV0dGjhwoVasmSJurq6wp7/wQcf6IorrlBTU5Pa29t16aWX6uqrr1ZHR8ekGw8AAJwvwxhr+9TOmzdPF110kTZs2OA/NmvWLF177bWqr6+P6Gecd955qq6u1sMPPxzR+V6vVwUFBerv75fH47HS3IgcGjqq89Yc663pXLdYea7smH8GAADpJtLnt6WekeHhYbW3t6uqqiroeFVVlXbt2hXRzxgdHdXAwIBOPvnkcc8ZGhqS1+sNesULO/UCAGAvS2Gkt7dXIyMjKiwsDDpeWFionp6eiH7GE088oUOHDunGG28c95z6+noVFBT4X8XFxVaaaQk1RgAAsFdUE1hD63AYYyKqzfH666/rkUceUWNjo0499dRxz6urq1N/f7//tW/fvmiaaRk1RgAASDxLkyOmTZumrKysMb0gBw8eHNNbEqqxsVHLli3T1q1btWjRognPdbvdcrvdVpoWtcAZM+QQAAASz1LPiMvlUllZmZqbm4OONzc3q7KyctzrXn/9dd1666167bXXdNVVV0XX0jhgvggAAPazvGyktrZWN998s8rLy1VRUaEXXnhBXV1dqqmpkXRsiOXLL7/Uq6++KulYEFm6dKmeeuopXXzxxf5elSlTpqigoCCGX8U65osAAGA/y2GkurpafX19Wrdunbq7uzVnzhw1NTWppKREktTd3R1Uc+T555/X0aNHdffdd+vuu+/2H7/lllv0yiuvTP4bxAjzRQAAsIflOiN2iFedkcHho5r9MPVFAACIh7jUGQEAAIg1wggAALAVYQQAANiKMAIAAGxFGAEAALYijAAAAFsRRgAAgK0IIwAAwFaEEQAAYCvCCAAAsFVah5HkL4QPAEDqS9swYozRDRtb7W4GAABpL23DyOEjI+rs9kqSZhd5NCUny+YWAQCQntI2jATaWlOhjIwMu5sBAEBaIoxIIocAAGAfwggAALAVYQQAANiKMAIAAGxFGAEAALYijAAAAFsRRgAAgK0IIwAAwFaEEQAAYCvCCAAAsBVhBAAA2IowAgAAbEUYAQAAtiKMAAAAWxFGAACArQgjAADAVoQRAABgK8IIAACwFWEEAADYijACAABsRRgB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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "model_train_val_eval(train_X_os,val_X,train_y_os,val_y,new_model)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "id": "2cab6208-3ea3-4947-981f-6e57b1d4e492",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>params</th>\n",
       "      <th>mean_test_score</th>\n",
       "      <th>rank_test_score</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>{'model': LogisticRegression(C=0.5, penalty=None), 'model__C': 0.5, 'model__penalty': 'l2'}</td>\n",
       "      <td>0.876066</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>{'model': LogisticRegression(C=0.5, penalty=None), 'model__C': 0.5, 'model__penalty': None}</td>\n",
       "      <td>0.876372</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>{'model': LogisticRegression(C=0.5, penalty=None), 'model__C': 3, 'model__penalty': 'l2'}</td>\n",
       "      <td>0.876327</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>{'model': LogisticRegression(C=0.5, penalty=None), 'model__C': 3, 'model__penalty': None}</td>\n",
       "      <td>0.876372</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>{'model': DecisionTreeClassifier(), 'model__max_depth': 3}</td>\n",
       "      <td>0.798773</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>{'model': DecisionTreeClassifier(), 'model__max_depth': 5}</td>\n",
       "      <td>0.848421</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                                                        params  \\\n",
       "0  {'model': LogisticRegression(C=0.5, penalty=None), 'model__C': 0.5, 'model__penalty': 'l2'}   \n",
       "1  {'model': LogisticRegression(C=0.5, penalty=None), 'model__C': 0.5, 'model__penalty': None}   \n",
       "2    {'model': LogisticRegression(C=0.5, penalty=None), 'model__C': 3, 'model__penalty': 'l2'}   \n",
       "3    {'model': LogisticRegression(C=0.5, penalty=None), 'model__C': 3, 'model__penalty': None}   \n",
       "4                                   {'model': DecisionTreeClassifier(), 'model__max_depth': 3}   \n",
       "5                                   {'model': DecisionTreeClassifier(), 'model__max_depth': 5}   \n",
       "\n",
       "   mean_test_score  rank_test_score  \n",
       "0         0.876066                4  \n",
       "1         0.876372                1  \n",
       "2         0.876327                3  \n",
       "3         0.876372                1  \n",
       "4         0.798773                6  \n",
       "5         0.848421                5  "
      ]
     },
     "execution_count": 90,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "res_df_2 = pd.DataFrame(grid_2.cv_results_,)\n",
    "pd.set_option('display.max_colwidth',100)\n",
    "res_df_2[['params','mean_test_score','rank_test_score']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "id": "f8ffe667-fb7d-40c2-bf5a-d2e97f1096a9",
   "metadata": {},
   "outputs": [],
   "source": [
    "sub['is_promoted'] = new_model.predict(test)\n",
    "sub.to_csv('PromoPredictionsub_2.csv',index=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "99dbff78-e951-472b-8a37-335307e53874",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
