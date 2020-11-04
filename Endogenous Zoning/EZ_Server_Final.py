from EZ_Model_Hard_Final import *
from mesa.visualization.UserParam import UserSettableParameter
import logging

def agent_portrayal(agent):
    if type(agent) is DevelopTag:
        return

    portrayal = {"Shape": "circle",
                 "Filled": "true",
                 "Layer": 0}

    if agent.type == 'renter':
        portrayal["Color"] = "red"
        portrayal["r"] = 0.8

    else:
        portrayal["Color"] = "blue"
        portrayal["r"] = 0.8

    return portrayal

grid = CanvasGrid(agent_portrayal, 20, 20, 600, 600)

chart_1 = ChartModule([{"Label": "Regulations",
                   "Color": "Black"},
                        {"Label": "Permits",
                        "Color": "Yellow"}],
                 data_collector_name='datacollector')

chart_2 = ChartModule([{"Label": "Renters",
                   "Color": "Red"},
                        {"Label": "Homeowners",
                        "Color": "Blue"}],
                 data_collector_name='datacollector')

chart_3 =  ChartModule([{"Label": "Home Affordability",
                   "Color": "Blue"},
                        {"Label": "Rent Affordability",
                        "Color": "Red"}],
                 data_collector_name='datacollector')

chart_4 = ChartModule([{"Label": "Percent Homeowner Payoff",
                   "Color": "Blue"},
                        {"Label": "Percent Renter Payoff",
                        "Color": "Red"}],
                 data_collector_name='datacollector')

chart_5 = ChartModule([{"Label": "Average Home Value",
                   "Color": "Orange"},
                        {"Label": "Average Agent Wealth",
                        "Color": "Yellow"}],
                 data_collector_name='datacollector')
#

chart_6 = ChartModule([{"Label": "Average Home Rent",
                   "Color": "Orange"},
                        {"Label": "Average Agent Budget",
                        "Color": "Yellow"}],
                 data_collector_name='datacollector')
#
# chart_3 = ChartModule([{"Label": "Homeowners",
#                    "Color": "Blue"}],
#                  data_collector_name='datacollector')

model_params = {
    "N": UserSettableParameter("slider", "agent_number", 200, 50, 500, 50),
    "D": UserSettableParameter("slider", "developed_property_number", 200, 50, 500, 50),
    "width": 20,
    "height": 20,
    "homeowner_renter_ratio": UserSettableParameter("slider", "homeowner_renter_ratio", 0.7, 0.1, 1.0, 0.1),
    "homeowner_renter_newagent_ratio": UserSettableParameter("slider", "homeowner_renter_newagent_ratio", 0.7, 0.1, 1.0, 0.1),
    "committee_size": UserSettableParameter("slider", "committee_size", 5, 2, 15, 1),
    "regulation_impact": UserSettableParameter("slider", "regulation_impact_percent", 0.15, 0.01, 0.30, 0.01),
    "population_growth_rate": UserSettableParameter("slider", "population_growth_rate", 0.01, 0.005, 0.1, 0.005)
}


server = ModularServer(ZoningModel,
                       [grid, chart_1, chart_2, chart_3, chart_4, chart_5, chart_6],
                       "Zoning Model", model_params)
server.port = 8522
server.launch()
