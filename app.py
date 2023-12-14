from shiny import App, Inputs, Outputs, Session, reactive, render, req, ui, run_app
import pandas as pd
import lightgbm as lgb
import shiny.experimental as x
import shap
import matplotlib.pyplot as plt
import math
import pickle
from functools import partial
import sklearn
import numpy as np
import io
from sklearn.metrics import r2_score

#Load & clean dataset
generalStats_pos = pd.read_csv(
    'E:/Inzynierka/generalStats_pos - generalStats_pos_cleaned.csv')
generalStats_pos = generalStats_pos.rename(columns={'G+A-PK': 'G_A_PK', 'Tkl+Int': 'Tkl_Int', 'G+A': 'G_A',
                                            'G-PK': 'G_PK', 'npxG+xAG': 'npxG_xAG', 'Def 3rd': 'Def_3rd', 'Mid 3rd': 'Mid_3rd', 'Att 3rd': 'Att_3rd'})

#load X train for linear models for shap
x_train_net = np.load('E:/Inzynierka/X_train_selected_net.npy')
x_train_lnr = np.load('E:/Inzynierka/X_train_selected_lnr.npy')
x_train_tests = {'linear regression': x_train_lnr, 'elasticNet': x_train_net}

#extract players and y
Players = generalStats_pos['Player']
y = generalStats_pos['value_log']
y.index = Players

#Load models
lgbmod = lgb.Booster(model_file='E:/Inzynierka/model/lightgbm.txt')
rfor = pickle.load(open('E:/Inzynierka/model/random_forest_model.pkl', 'rb'))
dtree = pickle.load(open('E:/Inzynierka/model/decision_tree_model.pkl', 'rb'))
enet = pickle.load(open('E:/Inzynierka/model/elasticnet.pkl', 'rb'))
lnr = pickle.load(open('E:/Inzynierka/model/linear_regression.pkl', 'rb'))

models = {'lightGBM': lgbmod, 'randomForest': rfor,
          'decisionTree': dtree, 'elasticNet': enet, 'linear regression': lnr}

#create feature of all models
featurenames = {
    'lightGBM': ['age', 'Nation_map', 'Team_map', '90s', 'PrgC', 'PrgP', 'PrgR', 'G_A_PK', 'Blocks', 'Int', 'Tkl_Int', 'Sh', 'Dist', 'SCA90'],
    'randomForest': ['age', 'Team_map', '90s', 'G_A', 'npxG_xAG', 'PrgP', 'PassLive'],
    'decisionTree': ['age', 'Team_map', 'G_A', 'PrgP'],
    'elasticNet': ['age', 'Nation_map', 'Team_map', 'PrgP', 'PrgR', 'Clr', 'Sh', 'Dist'],
    'linear regression': ['age', 'Nation_map', 'DF', 'FW', 'MF', 'Position_map', 'Leag_map', '90s', 'Gls', 'Ast', 'G_A', 'G_PK', 'npxG', 'xAG', 'npxG_xAG', 'PrgC', 'PrgP', 'G_A_PK', 'Tkl', 'TklW', 'Def_3rd', 'Mid_3rd', 'Att_3rd', 'Blocks', 'Tkl_Int', 'Err', 'Sh', 'SoT', 'Dist', 'FK', 'xG', 'SCA', 'SCA90', 'PassLive', 'GCA']
}

#array for dataframes
info = ['Player', 'age', 'Team_map', 'Position_map',
        'Nation_map', 'Leag_map', '90s', 'Gls', 'Ast']
#encoding categorical variables
league_mapping = {1: 'Premier League', 2: 'Serie A', 3: 'LaLiga'}
position_mapping = {1: 'FW', 2: 'MF', 3: 'DF', 4: 'FWMF', 5: 'DFMF', 6: 'DFFW'}
team_mapping = pd.read_csv(
    'E:/Inzynierka/mappng/generalStats_pos - Team mapping.csv')
nationality_mapping = pd.read_csv(
    'E:/Inzynierka/mappng/generalStats_pos - Nationality mapping.csv')

#frontend
app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.link(
            rel="stylesheet", href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css")
    ),
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.input_select(
                "player",
                "Choose a player:",
                Players.tolist()
            ),
            ui.input_select(
                "model",
                "Choose a model:",
                list(models.keys())
            ),
            ui.input_text(
                'marketvalue',
                'Enter value that is proposed for a player:',
                '0'
            ),
            ui.output_ui('featureSelection'),
            ui.input_action_button("show_nation_mapping", "Show nation mapping"),
            ui.div(
                ui.output_table("nation_mapping"),
                style="height: 200px; overflow-y: scroll;"
            ),
            ui.input_action_button("show_club_mapping", "Show club mapping"),
            ui.div(
                ui.output_table("club_mapping"),
                style="height: 200px; overflow-y: scroll;"
)
        ),
        ui.panel_main(
            ui.navset_tab(
                ui.nav(
                    'Individual predictions',
                    ui.div(
                        ui.h3('Player:'),
                        ui.output_data_frame('basicinfo'),
                        ui.br(),
                        ui.br(),
                        ui.h3('Statistics into the model:'),
                        ui.output_data_frame('modelStats'),
                        ui.br(),
                        ui.br(),
                        ui.h3('Estimation & Recommendation:'),
                        ui.h4(ui.output_text('predictValueOutInd')),
                        ui.output_ui('recommendationInd'),
                        ui.h6(ui.output_text('meanPredictValue')),
                        ui.br(),
                        ui.br(),
                        ui.h3('Analytics:'),
                        ui.h5('Descriptive statistics'),
                        ui.output_data_frame('stats'),
                        ui.br(),
                        ui.h5('How each stat influenced estimated value'),
                        ui.output_plot("waterfallInd", height='500px'),
                        ui.h5('Estimation vs real values'),
                        ui.output_ui('plotsInd')
                    ),
                    value='individual'
                ),
                ui.nav(
                    'Custom Predicitons',
                    ui.div(
                        ui.row(
                            ui.column(
                                4,
                                ui.output_ui('dynamic_input_fields')
                            ),
                            ui.column(
                                8,
                                ui.h3('Estimation & Recommendation:'),
                                ui.h4(ui.output_text('predictValueOutCust')),
                                ui.output_ui('recommendationCust'),
                                ui.br(),
                                ui.br(),
                                ui.h3('Analytics:'),
                                ui.h5('How each stat influenced estimated value'),
                                ui.output_plot("waterfallCust",
                                               height='500px'),
                                ui.br(),
                                ui.h5('Estimation vs real values'),
                                ui.output_ui('plotsCust')


                            )
                        ),
                    ),
                    value='custom',
                ),
                ui.nav(
                    'Model Dashboard',
                    ui.div(
                        ui.h5('Selected model estimations vs values collected from Transfermarkt'),
                        ui.input_select(
                            "modelCompared",
                            "Choose a model for comparison:",
                            list(models.keys())
                        ),
                        ui.output_text('modelPerformanceMetrics'),
                        ui.output_plot('modelComparisonPlot'),
                        ui.h5('Metrics of each model:'),
                        ui.output_plot('plot_parameters')
                    ),
                    value='modelDashboard',
                ),
                id='tabs'
            )
        )
    ),
    ui.tags.footer(ui.output_text('disclaimer'))
)

#backend
def server(input: Inputs, output: Outputs, session: Session):
#general functions
    def dataframe(): #returns X_test for the model that was chosen
        mod = input.model()
        X_test = generalStats_pos[featurenames[mod]]
        X_test.index = Players
        return X_test
    def playerData():
        player_data = dataframe()[dataframe().index == input.player()]
        return player_data

    def getCustomValues():
        feature_names = featurenames[input.model()]
        val = {features: float(input[features]())
               for features in feature_names}
        val = pd.DataFrame(val, index=[0])
        return val


    def predictValue(): #returns predicted value for given player
        if input.tabs() == 'individual':
            val = playerData()
        else:
            val = getCustomValues()
        val.rename(columns={'G_A': 'G+A', 'npxG_xAG': 'npxG+xAG'}, inplace=True)
        model = models[input.model()]
        estimatedval = model.predict(val)
        estimatedval = round(math.exp(estimatedval))
        return estimatedval

# Output for individual

    @output
    @render.text
    def meanPredictValue():
        estimatedval = 0
        mods = models.keys()

        for mod in mods:
            player = generalStats_pos.loc[generalStats_pos['Player'] == input.player(), featurenames[mod]]
            player.rename(columns={'G_A': 'G+A', 'npxG_xAG': 'npxG+xAG'}, inplace=True)
            print(player.columns)
            estimatedval = estimatedval + math.exp(models[mod].predict(
               player))
        estimatedval = estimatedval/len(mods)
        estimatedval = round(estimatedval)

        message = 'Mean predicted value of all models is: ' + \
            str(estimatedval)+'€'
        return message

    @output
    @render.text
    def predictValueOutInd():
        return "Estimated value for "+str(input.player()) + ": "+str(predictValue())+'€'+'('+str(input.model())+')'

    @render.text
    def recommendationtextInd():
        if (input.marketvalue() == ''):
            outcome = 'Enter value if you want to obtain recommendation'
        else:
            if (int(predictValue()) > int(input.marketvalue())):
                outcome = 'Proposed value is lower than the predicted value. That could mean player is UNDERVALUED'
            elif (int(predictValue()) < int(input.marketvalue())):
                outcome = 'Proposed value is higher than the predicted value. That could mean player is OVERVALUED'
            elif (int(predictValue()) < int(input.marketvalue())):
                outcome = 'The Player value is equal to the asking value.'
        return outcome


    @output
    @render.ui
    def featureSelection():
        mod = input.model()
        return ui.input_select('feature', 'Enter feature for scatter plot', featurenames[mod], multiple=True, selected=featurenames[mod][0])

    @output
    @render.ui
    def recommendationInd():
        if (int(predictValue()) > int(input.marketvalue())):
            return ui.h4(ui.output_text('recommendationtextInd'), style='color:green')
        elif (int(predictValue()) < int(input.marketvalue())):
            return ui.h4(ui.output_text('recommendationtextInd'), style='color:red')
        elif (int(predictValue()) < int(input.marketvalue())):
            return ui.h4(ui.output_text('recommendationtextInd'))

    @output
    @render.ui
    def plotsInd():
        heightplot = str(len(input.feature())*500)+'px'
        return ui.output_plot('scatterInd', width='100%', height=heightplot)

    @output
    @render.ui
    def dynamic_input_fields():
        model = input.model()
        feature_names = featurenames[model]
        df = playerData()
        feature_inputs = [ui.input_text(
            feature, 'Enter value for '+feature, str(df[feature][0])) for feature in feature_names]
        return ui.div(*feature_inputs)

    @output
    @render.data_frame
    def modelStats():
        df = playerData()
        return df

    @output
    @render.plot
    def scatterInd():
        player = input.player()
        player_data = playerData()
        value = math.log(predictValue())
        df = dataframe()
        feature = input.feature()
        numofplots = len(feature)
        fig, ax = plt.subplots(numofplots, 1)
        fig.set_size_inches(10, numofplots*10)
        if numofplots > 1:
            for i in range(0, numofplots):
                ax[i].scatter(df[feature[i]], y, color='blue',
                              label='All players')

                if not player_data.empty:
                    ax[i].scatter(player_data[feature[i]], value,
                                  color='red', label=player)

                ax[i].set_xlabel(feature[i])
                ax[i].set_ylabel('value_log')
                ax[i].set_title('Scatter plot with highlighted player')
                ax[i].legend()
        else:
            ax.scatter(df[feature[0]], y, color='blue', label='All players')
            if not player_data.empty:
                ax.scatter(player_data[feature[0]],
                           value, color='red', label=player)

            ax.set_xlabel(feature[0])
            ax.set_ylabel('value_log')
            ax.set_title('Scatter plot with highlighted player')
            ax.legend()
        return fig

    @output
    @render.plot
    def waterfallInd():
        plt.clf()
        mod = input.model()
        model = models[mod]
        if mod == 'elasticNet' or mod == 'linear regression':
            explainer = shap.Explainer(model, x_train_tests[mod])
        else:
            explainer = shap.Explainer(model)
        shap_values = explainer(playerData())
        fig = shap.plots.waterfall(shap_values[0], show=False)
        figx = plt.gcf()
        return figx

    @output
    @render.data_frame
    def basicinfo():
        player = input.player()
        df = generalStats_pos
        df = df[df['Player'] == player]

        merged_df = df.merge(
            nationality_mapping, left_on='Nation_map', right_on='Value', how='left')
        merged_df['Value'] = merged_df['Nationality'].fillna(
            merged_df['Value'])
        df = merged_df.drop(columns=['Nation_map', 'Value'])

        merged_df = df.merge(team_mapping, left_on='Team_map',
                             right_on='Value', how='left')
        merged_df['Value'] = merged_df['Club'].fillna(merged_df['Value'])
        df = merged_df.drop(columns=['Team_map', 'Value'])

        df['Position_map'][0] = position_mapping[df['Position_map'][0]]
        df = df.rename(columns={'Position_map': 'Position'})

        df['Leag_map'][0] = league_mapping[df['Leag_map'][0]]
        df = df.rename(columns={'Leag_map': 'League'})

        new_order = ['Player', 'age', 'Position',
                     'Club', 'League', 'Gls', 'Ast', '90s']
        df = df[new_order]
        return df

    @output
    @render.data_frame
    def stats():
        df = generalStats_pos
        player = input.player()
        player_data = df[df['Player'] == player].reset_index()
        stats = {'groups': ['age', 'Position_map', 'Team_map', 'Leag_map'], 'sum': [
        ], 'mean': [], 'median': [], 'max': [], 'min': [], 'std': [], 'q1': [], 'q3': []}
        for group in stats['groups']:
            grouped = df.groupby(group)['value'].sum()
            result = grouped[grouped.index == player_data[group][0]]
            stats['sum'].append(result)

            grouped = df.groupby(group)['value'].mean()
            result = grouped[grouped.index == player_data[group][0]]
            stats['mean'].append(result)

            grouped = df.groupby(group)['value'].median()
            result = grouped[grouped.index == player_data[group][0]]
            stats['median'].append(result)

            grouped = df.groupby(group)['value'].max()
            result = grouped[grouped.index == player_data[group][0]]
            stats['max'].append(result)

            grouped = df.groupby(group)['value'].min()
            result = grouped[grouped.index == player_data[group][0]]
            stats['min'].append(result)

            grouped = df.groupby(group)['value'].std()
            result = grouped[grouped.index == player_data[group][0]]
            stats['std'].append(result)

            grouped = df.groupby(group)['value'].quantile(0.25)
            result = grouped[grouped.index == player_data[group][0]]
            stats['q1'].append(result)

            grouped = df.groupby(group)['value'].quantile(0.75)
            result = grouped[grouped.index == player_data[group][0]]
            stats['q3'].append(result)

        result = df['value'].sum()
        stats['sum'].append(result)

        result = df['value'].mean()
        stats['mean'].append(result)

        result = df['value'].median()
        stats['median'].append(result)

        result = df['value'].max()
        stats['max'].append(result)

        result = df['value'].min()
        stats['min'].append(result)

        result = df['value'].std()
        stats['std'].append(result)

        result = df['value'].quantile(0.25)
        stats['q1'].append(result)

        result = df['value'].quantile(0.75)
        stats['q3'].append(result)

        stats['groups'].append('all')

        targetdf = pd.DataFrame(stats)

        return targetdf

# output for custom
    @render.text
    def recommendationtextCust():
        if (input.marketvalue() == ''):
            outcome = 'Enter value if you want to obtain recommendation'
        else:
            if (int(predictValue()) > int(input.marketvalue())):
                outcome = 'The Players asking value is lower than the predicted value. That could mean player is UNDERVALUED'
            elif (int(predictValue()) < int(input.marketvalue())):

                outcome = 'The Players asking value is higher than the predicted value. That could mean player is OVERVALUED'
            elif (int(predictValue()) < int(input.marketvalue())):
                outcome = 'The Player value is equal to the asking value'
        return outcome

    @output
    @render.ui
    def recommendationCust():
        if (int(predictValue()) > int(input.marketvalue())):
            return ui.h4(ui.output_text('recommendationtextCust'), style='color:green')
        elif (int(predictValue()) < int(input.marketvalue())):
            return ui.h4(ui.output_text('recommendationtextCust'), style='color:red')
        elif (int(predictValue()) < int(input.marketvalue())):
            return ui.h4(ui.output_text('recommendationtextCust'))

    @output
    @render.text
    def predictValueOutCust():
        return "Estimated value for "+str(input.player()) + ": "+str(predictValue())+'€'+'('+str(input.model())+')'

    @output
    @render.plot
    def waterfallCust():
        plt.clf()
        mod = input.model()
        model = models[mod]
        if mod == 'elasticNet' or mod == 'linear regression':
            explainer = shap.Explainer(model, x_train_tests[mod])
        else:
            explainer = shap.Explainer(model)
        shap_values = explainer(getCustomValues())
        fig = shap.plots.waterfall(shap_values[0], show=False)
        figx = plt.gcf()
        return figx

    @render.plot
    @reactive.event(input.tabs)
    def scatterCust():
        player = 'Custom Player'
        player_data = getCustomValues()
        value = math.log(predictValue())
        df = dataframe()
        feature = input.feature()
        numofplots = len(feature)
        fig, ax = plt.subplots(numofplots, 1)
        fig.set_size_inches(10, numofplots*10)
        if numofplots > 1:
            for i in range(0, numofplots):
                ax[i].scatter(df[feature[i]], y, color='blue',
                              label='All players')

                if not player_data.empty:
                    ax[i].scatter(player_data[feature[i]], value,
                                  color='red', label=player)

                ax[i].set_xlabel(feature[i])
                ax[i].set_ylabel('value_log')
                ax[i].set_title('Scatter plot with highlighted player')
                ax[i].legend()
        else:
            ax.scatter(df[feature[0]], y, color='blue', label='All players')
            if not player_data.empty:
                ax.scatter(player_data[feature[0]],
                           value, color='red', label=player)

            ax.set_xlabel(feature[0])
            ax.set_ylabel('value_log')
            ax.set_title('Scatter plot with highlighted player')
            ax.legend()
        return fig

    @output
    @render.ui
    def plotsCust():

        heightplot = str(len(input.feature())*500)+'px'
        return ui.output_plot('scatterCust', width='100%', height=heightplot)

    @output
    @render.plot
    def modelComparisonPlot():
        plt.clf()
        selectedModel = input.model()
        selectedModelCompared = input.modelCompared()
        modelCompared = models[selectedModelCompared]
        model = models[selectedModel]

        X_test = dataframe()
        X_test_compare = generalStats_pos[featurenames[selectedModelCompared]]
        X_test_compare = X_test_compare.rename(columns={'G_A': 'G+A', 'npxG_xAG': 'npxG+xAG'})
        X_test_compare.index = Players

        predictedValuesCompared = np.exp(modelCompared.predict(X_test_compare))
        actualValues = generalStats_pos['value']
        predictedValues = np.exp(model.predict(X_test))

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(actualValues, label='Actual Value', color='blue', linestyle = 'dashdot')
        ax.plot(predictedValues, label=selectedModel, color='orange', linestyle='solid', alpha=0.7)
        if model != modelCompared:
            ax.plot(predictedValuesCompared, label=selectedModelCompared, color='green', linestyle = 'dashed', alpha=0.4)
        plt.ylabel('value')
        plt.xlabel('players index')
        ax.legend()
        fig = plt.gcf()
        return fig



    @output
    @render.plot
    def plot_parameters():
        performance_params = {
            'Linear Regression': [0.67, 0.72, 0.85, 0.55, 0.53],
            'Elastic Net': [0.70, 0.80, 0.90, 0.49, 0.56],
            'Decision Tree': [0.66, 0.73, 0.86, 0.53, 0.55],
            'Random Forest': [0.56, 0.49, 0.70, 0.69, 0.48],
            'LightGBM': [0.50, 0.41, 0.64, 0.74, 0.41]
        }

        df = pd.DataFrame(performance_params, index=['MeanAE', 'MSE', 'RMSE', 'R^2', 'MedianAE'])
        df.columns = ['Linear Regression', 'Elastic Net', 'Decision Tree', 'Random Forest', 'LightGBM']

        df.plot(kind='bar', figsize=(10, 6))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title('Performance Parameters of Models')
        plt.ylabel('Values')
        plt.xlabel('Metrics')

        fig = plt.gcf()
        return fig


    @output
    @render.table
    def nation_mapping():
        if input.show_nation_mapping() % 2 != 0:
            nation_mapping = pd.DataFrame({"Nationality": ["United States of America", "Scotland", "England", "Morocco", "France", "Algeria", "Norway", "Switzerland", "Netherlands",
                                                      "Spain", "Argentina", "Brazil", "Paraguay", "Ghana", "Denmark", "Jamaica", "Nigeria", "Gabon", "Ivory Coast", "Grenada",
                                                      "Ireland", "Poland", "Bosnia and Herzegovina", "Germany", "Uruguay", "Mali", "Albania", "Wales", "Ecuador", "Croatia",
                                                      "Portugal", "Belgium", "Czech Republic", "Zambia", "Colombia", "Slovakia", "Sweden", "Egypt", "Northern Ireland", "Senegal",
                                                      "Iran", "Italy", "South Korea", "Mexico", "Democratic Republic of the Congo", "Guinea", "Serbia", "Cameroon", "Japan", "Ukraine",
                                                      "Costa Rica", "Burkina Faso", "Venezuela", "Austria", "Israel", "Australia", "Turkey", "Greece", "New Zealand", "Zimbabwe", "Bulgaria",
                                                      "The Gambia", "Guinea-Bissau", "Slovenia", "Tunisia", "Romania", "Martinique", "Iceland", "North Macedonia",
                                                      "Chile", "Lithuania", "Guadeloupe","Sierra Leone", "Cyprus", "Georgia", "Montenegro", "Russia", "Armenia",
                                                      "Angola", "Equatorial Guinea", "Finland", "Kosovo", "Uzbekistan", "Peru", "Togo", "Central African Republic", "Canada",
                                                      "Honduras", "Mozambique", "Dominican Republic"],
                                        "Mapping": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
                                                    "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40",
                                                    "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60",
                                                    "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "80",
                                                    "81", "82", "83", "84", "85", "86", "87", "88", "89", "90"]})
            return nation_mapping
        else:
            # Return empty DataFrame to hide the table
            return pd.DataFrame()
    @output
    @render.table
    def club_mapping():
        if input.show_club_mapping() % 2 != 0:
            club_mapping = pd.DataFrame({"Club": ["Leeds United", "Southampton", "Fulham", "West Ham", "Crystal Palace", "Wolves", "Brentford",
                                                "Manchester City", "Leicester City", "Liverpool", "Newcastle Utd", "Bournemouth", "Manchester Utd",
                                                "Chelsea", "Nott'ham Forest", "Aston Villa", "Everton", "Tottenham", "Brighton", "Arsenal",
                                                "Hellas Verona", "Roma", "Inter", "Milan", "Bologna", "Cremonese", "Spezia", "Torino", "Empoli",
                                                "Sassuolo", "Fiorentina", "Lazio", "Monza", "Udinese", "Lecce", "Sampdoria", "Juventus", "Atalanta",
                                                "Salernitana", "Napoli", "Betis", "Sevilla", "Athletic Club", "Valladolid", "Celta Vigo", "Almería",
                                                "Real Madrid", "Cádiz", "Barcelona", "Villarreal", "Getafe", "Real Sociedad", "Valencia", "Mallorca",
                                                "Girona", "Osasuna", "Elche", "Rayo Vallecano", "Espanyol", "Atlético Madrid"],
                                        "Mapping":[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                                    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
                                                    39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
                                                    57, 58, 59, 60]})
            return club_mapping
        else:
            # Return empty DataFrame to hide the table
            return pd.DataFrame()

    #Footer
    @output
    @render.text
    def disclaimer():
        return "Models were trained on the stastistics from season 2022/2023, outcome of the model may be outdated as of today. Data sources: FBREF, Transfermarkt"

app = App(app_ui, server)
