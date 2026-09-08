import unittest
from datetime import date, timedelta
from maritime_ai.data import load_freight_data, ports, vessels
from maritime_ai.database import BrokerTender
from maritime_ai.forecast import backtest, forecast
from maritime_ai.decision import feasibility, contract_options
from maritime_ai.tenders import CONTRACT_VOYAGES, as_utc, rank_tenders


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

    def test_tender_ranking_rejects_infeasible_and_expired_offers(self):
        today = date.today()
        good = BrokerTender(id=1, broker_name="Broker A", origin="Indonesia", destination_port="Paradip", cargo_tonnes=58000, cargo_type="Thermal coal",
                            laycan_start=as_utc(today), vessel="Supramax-58", contract="Spot (1 voyage)", voyages=1,
                            all_in_usd_per_tonne=24.5, valid_until=as_utc(today + timedelta(days=3)), notes="", status="SUBMITTED")
        expired = BrokerTender(id=2, broker_name="Broker B", origin="Indonesia", destination_port="Paradip", cargo_tonnes=58000, cargo_type="Thermal coal",
                               laycan_start=as_utc(today), vessel="Supramax-58", contract="Spot (1 voyage)", voyages=1,
                               all_in_usd_per_tonne=20, valid_until=as_utc(today - timedelta(days=1)), notes="", status="SUBMITTED")
        ranking = rank_tenders([good, expired], "Paradip", 58000)
        self.assertEqual(CONTRACT_VOYAGES["6-voyage contract"], 6)
        self.assertEqual(ranking.iloc[0]["broker"], "Broker A")
        self.assertTrue(ranking.iloc[0]["eligible"])
        self.assertFalse(ranking.iloc[1]["eligible"])

    def test_thermal_coal_tender_rejects_capesize_at_paradip_reference_berths(self):
        today = date.today()
        capesize_offer = BrokerTender(id=3, broker_name="Broker C", origin="Indonesia", destination_port="Paradip", cargo_tonnes=58000, cargo_type="Thermal coal",
                                      laycan_start=as_utc(today), vessel="Capesize-180", contract="Spot (1 voyage)", voyages=1,
                                      all_in_usd_per_tonne=19.0, valid_until=as_utc(today + timedelta(days=3)), notes="", status="SUBMITTED")
        ranking = rank_tenders([capesize_offer], "Paradip", 58000, "Thermal coal")
        self.assertFalse(ranking.iloc[0]["eligible"])
        self.assertIn("No compatible cargo berth", ranking.iloc[0]["reason"])

    def test_shortlisted_tender_remains_eligible_for_award(self):
        today = date.today()
        tender = BrokerTender(id=4, broker_name="Broker D", origin="Indonesia", destination_port="Paradip", cargo_tonnes=58000, cargo_type="Thermal coal",
                              laycan_start=as_utc(today), vessel="Supramax-58", contract="Spot (1 voyage)", voyages=1,
                              all_in_usd_per_tonne=24.0, valid_until=as_utc(today + timedelta(days=3)), notes="", status="SHORTLISTED")
        ranking = rank_tenders([tender], "Paradip", 58000, "Thermal coal")
        self.assertTrue(ranking.iloc[0]["eligible"])
