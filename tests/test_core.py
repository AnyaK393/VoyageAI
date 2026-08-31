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
