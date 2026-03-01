
import numpy as np

class YieldPredictor:
    def __init__(self):
        # Base yields in Tons per hectare for Tamil Nadu context
        # Reference: TN Agricultural University (TNAU) averages
        self.base_yields = {
            "maize": 5.5,    # Higher in TN high-yield zones
            "wheat": 3.0,    # Less common in TN, but niche
            "tomato": 30.0,  # High production in areas like Krishnagiri/Coimbatore
            "paddy": 4.5
        }
    
    def predict(self, data):
        """
        Predict yield based on input data for multiple crops with Advanced Metrics
        """
        crops = data.get('crops', ['maize'])
        if isinstance(crops, str): crops = [crops]
        
        area = float(data.get('area', 1.0))
        rainfall = float(data.get('rainfall', 1000))
        temp = float(data.get('temperature', 28))
        humidity = float(data.get('humidity', 65))
        n = float(data.get('fertilizer_n', 100))
        p = float(data.get('fertilizer_p', 50))
        k = float(data.get('fertilizer_k', 50))
        ph = float(data.get('soil_ph', 6.5))
        om = float(data.get('organic_matter', 2.0))
        soil_type = data.get('soil_type', 'loam').lower()
        
        results = []
        
        # Soil type factor
        soil_factors = {"loam": 1.1, "clay": 0.95, "sandy": 0.8, "silt": 1.05}
        soil_factor = soil_factors.get(soil_type, 1.0)

        # Organic matter factor
        om_factor = 0.8 + (min(om, 5.0) / 10.0) 

        # Base Costs per Acre (INR) - Mocked for Indian context
        crop_costs = {
            "maize": 25000,
            "wheat": 22000,
            "tomato": 65000
        }

        for crop_type in crops:
            crop_type = crop_type.lower()
            yield_per_hectare = self.base_yields.get(crop_type, 2.5)
            
            # Multipliers
            rainfall_factor = 1.0 - abs(rainfall - 950) / 2500
            temp_factor = 1.0 - abs(temp - 30) / 60
            humidity_factor = 1.0 - abs(humidity - 60) / 150
            ph_factor = 1.0 - abs(ph - 6.8) / 12
            fert_factor = (n/120 + p/60 + k/60) / 3.0
            
            multiplier = rainfall_factor * temp_factor * ph_factor * fert_factor * soil_factor * om_factor * humidity_factor
            
            final_yield_tons = yield_per_hectare * (area / 2.471) * multiplier
            final_yield_tons = max(0.1, final_yield_tons)
            
            # Financials
            market_price = (22000 if crop_type == 'maize' else 35000 if crop_type == 'wheat' else 15000)
            revenue = final_yield_tons * market_price
            total_cost = area * crop_costs.get(crop_type, 30000)
            profit = revenue - total_cost
            roi = (profit / total_cost) * 100 if total_cost > 0 else 0
            break_even_yield = total_cost / market_price if market_price > 0 else 0

            results.append({
                "crop_type": crop_type,
                "predicted_yield_tons": round(final_yield_tons, 2),
                "yield_per_acre": round(final_yield_tons / area, 2),
                "market_price_est": round(revenue, 2),
                "roi": round(roi, 1),
                "profit": round(profit, 2),
                "break_even_tons": round(break_even_yield, 2)
            })
        
        # Risk assessment
        risk_score = 0
        if rainfall < 500 or rainfall > 2000: risk_score += 3
        if temp > 38 or temp < 15: risk_score += 3
        if ph < 5.5 or ph > 8.0: risk_score += 2
        
        risk_level = "LOW"
        if risk_score >= 5: risk_level = "HIGH"
        elif risk_score >= 3: risk_level = "MEDIUM"

        # Insights
        insights = []
        if n < 80: insights.append("Nutrient Deficit: Increase Nitrogen by 20% for optimal growth.")
        if ph < 6: insights.append("Soil Acidity: Apply 500kg/acre of Lime to normalize pH.")
        if rainfall < 700: insights.append("Water Stress: Supplement with 2 extra irrigation cycles.")
        if om < 1.5: insights.append("Soil Health: Integrate green manure or compost.")
        if not insights: insights.append("Ideal conditions: Maintain current fertilization schedule.")

        return {
            "predictions": results,
            "total_estimated_value": sum(r['market_price_est'] for r in results),
            "total_profit": sum(r['profit'] for r in results),
            "suitability_score": round(max(0, min(100, multiplier * 100)), 1),
            "risk_level": risk_level,
            "insights": insights[:3],
            "factors": {
                "rainfall": round(rainfall_factor, 2),
                "temperature": round(temp_factor, 2),
                "humidity": round(humidity_factor, 2),
                "ph": round(ph_factor, 2),
                "nutrients": round(fert_factor, 2),
                "soil": round(soil_factor, 2),
                "organic": round(om_factor, 2)
            },
            "charts": {
                "historical": [round(final_yield_tons * x, 2) for x in [0.85, 0.92, 1.05, 0.98, 1.0]],
                "nutrient_balance": [n, p, k],
                "seasonal_trend": [60, 75, 90, 85, 70]
            }
        }
