import pandas as pd
import sqlite3
import math
from typing import Dict, Tuple
import os

class SafeGraphPropertyLookup:
    def __init__(self, auto_setup=True):
        self.db_path = 'property_lookup.db'
        if auto_setup:
            self.setup_database()
        
    def setup_database(self):
        """
        One-time processing of SafeGraph data into fast lookup
        Run this once after downloading SafeGraph files
        """
        
        print("Setting up property lookup database...")
        
        # Check if files exist
        required_files = ['cbg_b25.csv', 'cbg_geographic_data.csv']
        missing_files = []
        
        for file in required_files:
            try:
                with open(file, 'r') as f:
                    pass
            except FileNotFoundError:
                missing_files.append(file)
        
        if missing_files:
            print("❌ ERROR: Missing required SafeGraph files:")
            for file in missing_files:
                print(f"   - {file}")
            print("\n📥 Please download SafeGraph Open Census Data first:")
            print("   1. Go to: https://www.safegraph.com/free-data/open-census-data")
            print("   2. Register (free) and download the 2020 5-year ACS data")
            print("   3. Extract these files to the same directory as this script:")
            for file in required_files:
                print(f"      - {file}")
            print("   4. Run this script again")
            return False
        
        print("✅ Found required files, processing...")
        
        try:
            # Load housing data (B25 tables contain housing info)
            print("Loading housing data...")
            housing_df = pd.read_csv('cbg_b25.csv')
            print(f"   Loaded {len(housing_df):,} records from cbg_b25.csv")
            
            # Load geographic centroids
            print("Loading geographic data...")
            geo_df = pd.read_csv('cbg_geographic_data.csv')
            print(f"   Loaded {len(geo_df):,} records from cbg_geographic_data.csv")
            
        except Exception as e:
            print(f"❌ ERROR loading CSV files: {e}")
            return False
        
        # Merge housing data with geographic coordinates
        print("Merging housing and geographic data...")
        combined_df = housing_df.merge(
            geo_df[['census_block_group', 'latitude', 'longitude']], 
            on='census_block_group'
        )
        
        # Extract housing units and median values
        # B25001e1 = Total housing units
        # B25077e1 = Median home value
        required_columns = ['census_block_group', 'B25001e1', 'B25077e1']
        missing_columns = [col for col in required_columns if col not in housing_df.columns]
        
        if missing_columns:
            print(f"❌ ERROR: Missing required columns in housing data: {missing_columns}")
            print("Available columns:", list(housing_df.columns)[:10], "...")
            return False
            
        combined_df = combined_df[[
            'census_block_group',
            'latitude', 
            'longitude',
            'B25001e1',  # Housing units
            'B25077e1'   # Median home value
        ]].rename(columns={
            'B25001e1': 'housing_units',
            'B25077e1': 'median_value'
        })
        
        # Clean data
        print("Cleaning data...")
        initial_count = len(combined_df)
        combined_df = combined_df.dropna()
        combined_df['housing_units'] = combined_df['housing_units'].astype(int)
        combined_df['median_value'] = combined_df['median_value'].fillna(0).astype(int)
        
        print(f"   Cleaned data: {len(combined_df):,} records (removed {initial_count - len(combined_df):,} invalid records)")
        
        # Create SQLite database
        print("Creating SQLite database...")
        conn = sqlite3.connect(self.db_path)
        
        # Insert data directly (pandas will create the table automatically)
        combined_df.to_sql('block_groups', conn, if_exists='replace', index=False)
        
        # Create spatial index after table is created
        conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_location ON block_groups (latitude, longitude)
        ''')
        
        conn.close()
        print(f"✅ Database created successfully: {self.db_path}")
        print(f"   Ready to analyze {len(combined_df):,} Census Block Groups")
        return True
        
    def analyze_location(self, lat: float, lon: float, radius_miles: float = 0.5) -> Dict:
        """
        Pure intelligence analysis - no filtering, just report the data
        Default to 0.5-mile radius to target 5K-10K homes per location
        """
        
        conn = sqlite3.connect(self.db_path)
        
        # Calculate bounding box for efficiency
        lat_delta = radius_miles / 69.0  # degrees
        lon_delta = radius_miles / (69.0 * math.cos(math.radians(lat)))
        
        # Query block groups in bounding box
        query = '''
            SELECT latitude, longitude, housing_units, median_value
            FROM block_groups
            WHERE latitude BETWEEN ? AND ?
            AND longitude BETWEEN ? AND ?
        '''
        
        cursor = conn.execute(query, (
            lat - lat_delta, lat + lat_delta,
            lon - lon_delta, lon + lon_delta
        ))
        
        results = cursor.fetchall()
        conn.close()
        
        # Aggregate all data within radius - no filtering
        total_units = 0
        weighted_value_sum = 0
        total_weight = 0
        block_groups_analyzed = 0
        
        for bg_lat, bg_lon, units, value in results:
            distance = self.haversine_distance(lat, lon, bg_lat, bg_lon)
            
            if distance <= radius_miles and units > 0:
                block_groups_analyzed += 1
                total_units += units
                if value > 0:
                    weighted_value_sum += value * units
                    total_weight += units
        
        # Calculate average value
        avg_value = int(weighted_value_sum / total_weight) if total_weight > 0 else 0
        
        # Just return the intelligence - no qualification filtering
        return {
            'home_count': total_units,
            'avg_home_value': avg_value,
            'radius_miles': radius_miles,
            'block_groups_analyzed': block_groups_analyzed,
            'total_area_sq_miles': round(3.14159 * (radius_miles ** 2), 2),
            'homes_per_sq_mile': round(total_units / (3.14159 * (radius_miles ** 2)), 1) if total_units > 0 else 0,
            'data_source': 'census_block_groups'
        }
    
    def get_market_intelligence(self, lat: float, lon: float, radius_miles: float = 0.5) -> Dict:
        """
        Get market intelligence for business decision making
        This adds some basic market analysis on top of raw data
        """
        
        base_data = self.analyze_location(lat, lon, radius_miles)
        
        # Add market analysis
        revenue_potential = base_data['home_count'] * base_data['avg_home_value'] * 0.015  # 1.5% of home value
        
        # Market classification
        density = base_data['homes_per_sq_mile']
        if density > 2000:
            market_type = "Dense Urban"
        elif density > 800:
            market_type = "Urban"
        elif density > 300:
            market_type = "Suburban"
        elif density > 50:
            market_type = "Rural Suburban"
        else:
            market_type = "Rural"
            
        # Market attractiveness (for your business - targeting 5K-10K homes)
        home_count = base_data['home_count']
        avg_value = base_data['avg_home_value']
        
        if 5000 <= home_count <= 12000 and avg_value > 250000:
            market_attractiveness = "High"
        elif 3000 <= home_count <= 15000 and avg_value > 200000:
            market_attractiveness = "Medium"
        elif 1000 <= home_count <= 20000 and avg_value > 150000:
            market_attractiveness = "Low"
        else:
            market_attractiveness = "Poor"
        
        return {
            **base_data,
            'revenue_potential': revenue_potential,
            'market_type': market_type,
            'market_attractiveness': market_attractiveness
        }
    
    def get_optimal_radius_for_target(self, lat: float, lon: float, target_homes: int = 7500) -> float:
        """
        Find the optimal radius to get approximately target_homes
        Useful for adaptive sizing based on area density
        """
        
        # Test different radius sizes to find best fit
        radii_to_test = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
        best_radius = 0.5
        best_diff = float('inf')
        
        for radius in radii_to_test:
            result = self.analyze_location(lat, lon, radius)
            diff = abs(result['home_count'] - target_homes)
            
            if diff < best_diff:
                best_diff = diff
                best_radius = radius
        
        return best_radius
    
    def haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in miles"""
        R = 3959  # Earth radius in miles
        
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    def batch_analyze(self, locations: list) -> Dict:
        """
        Process all 60 locations efficiently
        Target: <3 seconds for all locations
        """
        
        results = {}
        
        for i, location in enumerate(locations):
            lat, lon = location['lat'], location['lon']
            result = self.analyze_location(lat, lon)
            results[f"location_{i}"] = result
            
        return results


def test_lookup():
    """Test the lookup system with a sample location"""
    
    print("\n🧪 Testing property lookup...")
    
    try:
        lookup = SafeGraphPropertyLookup(auto_setup=False)
        
        # Test pure intelligence gathering (no filtering)
        print(f"\n📊 Pure Intelligence Analysis:")
        
        # Test different radius sizes to find right target for NYC
        for radius in [0.25, 0.5, 0.75]:
            print(f"\n   📍 NYC - {radius} mile radius:")
            nyc_intel = lookup.analyze_location(40.7128, -74.0060, radius)
            nyc_market = lookup.get_market_intelligence(40.7128, -74.0060, radius)
            print(f"      Homes: {nyc_intel['home_count']:,}")
            print(f"      Avg Value: ${nyc_intel['avg_home_value']:,}")
            print(f"      Density: {nyc_intel['homes_per_sq_mile']:,} homes/sq mi")
            print(f"      Market Type: {nyc_market['market_type']}")
            print(f"      Revenue Potential: ${nyc_market['revenue_potential']:,}")
            
        # Test suburban area with target radius
        print(f"\n   📍 Denver Suburbs - 0.5 mile radius:")
        denver_intel = lookup.analyze_location(39.7391, -104.9847, 0.5)
        denver_market = lookup.get_market_intelligence(39.7391, -104.9847, 0.5)
        print(f"      Homes: {denver_intel['home_count']:,}")
        print(f"      Avg Value: ${denver_intel['avg_home_value']:,}")
        print(f"      Revenue Potential: ${denver_market['revenue_potential']:,}")
        print(f"      Market Type: {denver_market['market_type']}")
        print(f"      Attractiveness: {denver_market['market_attractiveness']}")
        
        # Test adaptive radius feature
        print(f"\n   🎯 Adaptive Radius Test (targeting 7,500 homes):")
        optimal_radius = lookup.get_optimal_radius_for_target(39.7391, -104.9847, 7500)
        adaptive_result = lookup.analyze_location(39.7391, -104.9847, optimal_radius)
        print(f"      Optimal radius: {optimal_radius} miles")
        print(f"      Homes found: {adaptive_result['home_count']:,}")
        print(f"      Target hit: {abs(adaptive_result['home_count'] - 7500) < 1000}")
        
        print(f"\n✅ Intelligence system working! Perfect for 5K-10K home targeting.")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🏠 SafeGraph Property Lookup Setup")
    print("=" * 40)
    
    # Try to set up the database
    lookup = SafeGraphPropertyLookup()
    
    # If setup succeeded, run a test
    if os.path.exists('property_lookup.db'):
        test_lookup()