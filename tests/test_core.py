import unittest
from maritime_ai.data import load_freight_data, ports, vessels
from maritime_ai.forecast import backtest, forecast
from maritime_ai.decision import feasibility, contract_options


class CoreTest(unittest.TestCase):
    def test_forecast_and_backtest_are_runnable(self):
        data, _ = load_freight_data()
        result = backtest(data)
        future = forecast(data, 6)
        self.assertGreater(result["n_predictions"], 10)
        self.assertGreater(result["mae"], 0)
        self.assertEqual(len(future), 6)
        self.assertTrue(future.rate_usd_per_tonne.notna().all())


    def test_haldia_rejects_capesize_and_contract_options_return_three_rows(self):
        port = ports().set_index("port").loc["Haldia"]
        cape = vessels().query("vessel == 'Capesize-180'").iloc[0]
        self.assertFalse(feasibility(port, cape, 58000)["feasible"])
        data, _ = load_freight_data()
        options = contract_options(forecast(data, 12), vessels().iloc[0], 58000)
        self.assertEqual(len(options), 3)

    def test_bunker_scenario_changes_forecast_and_contract_costs(self):
        data, _ = load_freight_data()
        vessel = vessels().iloc[0]
        base_forecast = forecast(data, 12)
        stressed_forecast = forecast(data, 12, {"bunker_pct": 15})
        self.assertFalse(base_forecast["bunker_usd_per_tonne"].equals(stressed_forecast["bunker_usd_per_tonne"]))
        base_options = contract_options(base_forecast, vessel, 58000)
        stressed_options = contract_options(stressed_forecast, vessel, 58000, scenario={"bunker_pct": 15})
        self.assertGreater(stressed_options["all_in_usd_per_tonne"].min(), base_options["all_in_usd_per_tonne"].min())
        self.assertGreater(stressed_options["risk_allowance_usd"].max(), 0)
