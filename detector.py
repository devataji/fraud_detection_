#!/usr/bin/env python3
"""
XGBoost Model Tester for APK Analysis
-------------------------------------
Tests a trained XGBoost model using APK analysis data from JSON/CSV files.
Handles feature engineering, preprocessing, and generates predictions with confidence scores.

USAGE
-----
python model_tester.py --model model.pkl --data analysis_report.json [--csv frequencies.csv] [--output results.json]

DEPENDENCIES
------------
- Python 3.7+
- xgboost
- pandas
- numpy
- scikit-learn
- joblib (for model loading)
"""

import argparse
import json
import pandas as pd
import numpy as np
import pickle
import joblib
import sys
import os
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
except ImportError:
    print("[!] XGBoost not installed. Install with: pip install xgboost")
    sys.exit(1)

try:
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    print("[!] scikit-learn not installed. Install with: pip install scikit-learn")
    sys.exit(1)


class APKFeatureExtractor:
    """Extract and engineer features from APK analysis data"""
    
    def __init__(self):
        self.permission_risk_scores = {
            'android.permission.INTERNET': 3,
            'android.permission.ACCESS_NETWORK_STATE': 2,
            'android.permission.ACCESS_FINE_LOCATION': 4,
            'android.permission.ACCESS_COARSE_LOCATION': 3,
            'android.permission.CAMERA': 3,
            'android.permission.RECORD_AUDIO': 4,
            'android.permission.READ_CONTACTS': 4,
            'android.permission.READ_SMS': 5,
            'android.permission.SEND_SMS': 5,
            'android.permission.CALL_PHONE': 4,
            'android.permission.READ_PHONE_STATE': 3,
            'android.permission.WRITE_EXTERNAL_STORAGE': 3,
            'android.permission.READ_EXTERNAL_STORAGE': 2,
            'android.permission.WAKE_LOCK': 2,
            'android.permission.RECEIVE_BOOT_COMPLETED': 3,
            'android.permission.SYSTEM_ALERT_WINDOW': 4,
            'android.permission.GET_ACCOUNTS': 3,
            'android.permission.BIND_DEVICE_ADMIN': 5,
            'android.permission.INSTALL_PACKAGES': 5,
            'android.permission.DELETE_PACKAGES': 5,
        }
        
        self.dangerous_keywords = [
            'admin', 'root', 'su', 'shell', 'exec', 'runtime', 'process',
            'install', 'uninstall', 'delete', 'remove', 'hide', 'stealth',
            'bypass', 'crack', 'hack', 'exploit', 'payload', 'trojan',
            'malware', 'virus', 'spy', 'keylog', 'steal', 'phish'
        ]
    
    def extract_features(self, json_data: Dict, csv_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Extract comprehensive features from APK analysis data"""
        features = {}
        
        # Manifest features
        manifest = json_data.get('manifest', {})
        features.update(self._extract_manifest_features(manifest))
        
        # Binder analysis features
        binder_static = json_data.get('binder_static', {})
        features.update(self._extract_binder_features(binder_static))
        
        # Dynamic analysis features (if available)
        if json_data.get('strace'):
            features.update(self._extract_strace_features(json_data['strace']))
        if json_data.get('binder_trace'):
            features.update(self._extract_binder_trace_features(json_data['binder_trace']))
            
        # CSV frequency features (if available)
        if csv_data is not None:
            features.update(self._extract_csv_features(csv_data))
            
        return features
    
    def _extract_manifest_features(self, manifest: Dict) -> Dict[str, Any]:
        """Extract features from manifest data"""
        features = {}
        
        # Basic info
        features['version_code'] = int(manifest.get('version_code', 0))
        features['min_sdk'] = int(manifest.get('min_sdk', 0))
        features['target_sdk'] = int(manifest.get('target_sdk', 0))
        features['sdk_gap'] = features['target_sdk'] - features['min_sdk']
        
        # Component counts
        features['num_activities'] = len(manifest.get('activities', []))
        features['num_receivers'] = len(manifest.get('receivers', []))
        features['num_services'] = len(manifest.get('services', []))
        features['num_providers'] = len(manifest.get('providers', []))
        features['total_components'] = (features['num_activities'] + features['num_receivers'] + 
                                      features['num_services'] + features['num_providers'])
        
        # Permission analysis
        permissions = manifest.get('permissions', [])
        features['num_permissions'] = len(permissions)
        features['permission_risk_score'] = sum(
            self.permission_risk_scores.get(perm, 1) for perm in permissions
        )
        features['avg_permission_risk'] = (features['permission_risk_score'] / 
                                         max(features['num_permissions'], 1))
        
        # Dangerous permissions
        dangerous_perms = [p for p in permissions if self.permission_risk_scores.get(p, 0) >= 4]
        features['num_dangerous_permissions'] = len(dangerous_perms)
        features['has_dangerous_permissions'] = int(len(dangerous_perms) > 0)
        
        # Specific permission flags
        features['has_internet'] = int('android.permission.INTERNET' in permissions)
        features['has_location'] = int(any('LOCATION' in p for p in permissions))
        features['has_sms'] = int(any('SMS' in p for p in permissions))
        features['has_phone'] = int(any('PHONE' in p for p in permissions))
        features['has_camera'] = int('android.permission.CAMERA' in permissions)
        features['has_contacts'] = int('android.permission.READ_CONTACTS' in permissions)
        features['has_storage'] = int(any('STORAGE' in p for p in permissions))
        features['has_admin'] = int('android.permission.BIND_DEVICE_ADMIN' in permissions)
        
        # Package name analysis
        package_name = manifest.get('package', '').lower()
        features['package_length'] = len(package_name)
        features['package_parts'] = len(package_name.split('.'))
        features['has_suspicious_keywords'] = int(any(keyword in package_name 
                                                    for keyword in self.dangerous_keywords))
        
        # Component name analysis
        all_components = (manifest.get('activities', []) + manifest.get('receivers', []) +
                         manifest.get('services', []) + manifest.get('providers', []))
        component_text = ' '.join(all_components).lower()
        features['component_suspicious_score'] = sum(1 for keyword in self.dangerous_keywords 
                                                   if keyword in component_text)
        
        return features
    
    def _extract_binder_features(self, binder_static: Dict) -> Dict[str, Any]:
        """Extract features from static binder analysis"""
        features = {}
        
        features['total_invoke'] = binder_static.get('total_invoke', 0)
        features['binder_like_invoke'] = binder_static.get('binder_like_invoke', 0)
        features['binder_ratio'] = (features['binder_like_invoke'] / 
                                  max(features['total_invoke'], 1))
        
        # Method and class diversity
        calls_by_method = binder_static.get('calls_by_method', {})
        calls_by_class = binder_static.get('calls_by_class', {})
        
        features['unique_binder_methods'] = len(calls_by_method)
        features['unique_binder_classes'] = len(calls_by_class)
        
        if calls_by_method:
            method_counts = list(calls_by_method.values())
            features['max_method_calls'] = max(method_counts)
            features['avg_method_calls'] = np.mean(method_counts)
            features['method_call_std'] = np.std(method_counts)
        else:
            features['max_method_calls'] = 0
            features['avg_method_calls'] = 0
            features['method_call_std'] = 0
            
        return features
    
    def _extract_strace_features(self, strace_data: Dict) -> Dict[str, Any]:
        """Extract features from strace data"""
        features = {}
        
        counts = strace_data.get('counts', {})
        total_calls = strace_data.get('total', 0)
        
        features['total_syscalls'] = total_calls
        features['unique_syscalls'] = len(counts)
        
        if counts:
            call_values = list(counts.values())
            features['max_syscall_count'] = max(call_values)
            features['avg_syscall_count'] = np.mean(call_values)
            features['syscall_entropy'] = self._calculate_entropy(call_values)
            
            # Specific syscall categories
            network_calls = sum(v for k, v in counts.items() 
                              if k in ['connect', 'socket', 'send', 'recv', 'sendto', 'recvfrom'])
            file_calls = sum(v for k, v in counts.items() 
                           if k in ['open', 'read', 'write', 'close', 'unlink', 'rename'])
            process_calls = sum(v for k, v in counts.items() 
                              if k in ['fork', 'exec', 'clone', 'kill', 'waitpid'])
            
            features['network_syscall_ratio'] = network_calls / max(total_calls, 1)
            features['file_syscall_ratio'] = file_calls / max(total_calls, 1)
            features['process_syscall_ratio'] = process_calls / max(total_calls, 1)
        else:
            features.update({
                'max_syscall_count': 0, 'avg_syscall_count': 0, 'syscall_entropy': 0,
                'network_syscall_ratio': 0, 'file_syscall_ratio': 0, 'process_syscall_ratio': 0
            })
            
        return features
    
    def _extract_binder_trace_features(self, binder_trace_data: Dict) -> Dict[str, Any]:
        """Extract features from binder trace data"""
        features = {}
        
        counts = binder_trace_data.get('counts', {})
        total_events = binder_trace_data.get('total', 0)
        
        features['total_binder_events'] = total_events
        features['unique_binder_events'] = len(counts)
        
        if counts:
            event_values = list(counts.values())
            features['max_binder_event_count'] = max(event_values)
            features['avg_binder_event_count'] = np.mean(event_values)
            features['binder_event_entropy'] = self._calculate_entropy(event_values)
        else:
            features.update({
                'max_binder_event_count': 0, 'avg_binder_event_count': 0, 'binder_event_entropy': 0
            })
            
        return features
    
    def _extract_csv_features(self, csv_data: pd.DataFrame) -> Dict[str, Any]:
        """Extract features from CSV frequency data"""
        features = {}
        
        if csv_data.empty:
            return features
            
        # Group by category
        for category in csv_data['Category'].unique():
            cat_data = csv_data[csv_data['Category'] == category]
            prefix = category.lower().replace('_', '')
            
            features[f'{prefix}_total_count'] = cat_data['Count'].sum()
            features[f'{prefix}_unique_items'] = len(cat_data)
            features[f'{prefix}_max_count'] = cat_data['Count'].max()
            features[f'{prefix}_avg_frequency'] = cat_data['Frequency'].mean()
            features[f'{prefix}_entropy'] = self._calculate_entropy(cat_data['Count'].values)
            
        return features
    
    def _calculate_entropy(self, values: List[float]) -> float:
        """Calculate entropy of a distribution"""
        if not values or sum(values) == 0:
            return 0.0
        probabilities = np.array(values) / sum(values)
        probabilities = probabilities[probabilities > 0]  # Remove zeros
        return -sum(probabilities * np.log2(probabilities))


class ModelTester:
    """Test XGBoost model with APK analysis data"""
    
    def __init__(self, model_path: str):
        """Load the trained model"""
        self.model = self._load_model(model_path)
        self.feature_extractor = APKFeatureExtractor()
        self.feature_names = None
        
    def _load_model(self, model_path: str):
        """Load model from pickle file"""
        try:
            # Try joblib first
            return joblib.load(model_path)
        except:
            try:
                # Try pickle
                with open(model_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                raise RuntimeError(f"Failed to load model from {model_path}: {e}")
    
    def test_model(self, json_path: str, csv_path: Optional[str] = None) -> Dict[str, Any]:
        """Test the model with APK analysis data"""
        
        # Load data
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            
        csv_data = None
        if csv_path and os.path.exists(csv_path):
            csv_data = pd.read_csv(csv_path)
            
        # Extract features
        features = self.feature_extractor.extract_features(json_data, csv_data)
        
        # Prepare feature vector
        feature_df = pd.DataFrame([features])
        
        # Handle missing features (fill with 0)
        if hasattr(self.model, 'feature_names_in_'):
            expected_features = self.model.feature_names_in_
            for col in expected_features:
                if col not in feature_df.columns:
                    feature_df[col] = 0
            feature_df = feature_df[expected_features]
        elif hasattr(self.model, 'get_booster'):
            # XGBoost model
            try:
                expected_features = self.model.get_booster().feature_names
                if expected_features:
                    for col in expected_features:
                        if col not in feature_df.columns:
                            feature_df[col] = 0
                    feature_df = feature_df[expected_features]
            except:
                pass
        
        # Make predictions
        try:
            # Get prediction probabilities if available
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(feature_df)[0]
                prediction = self.model.predict(feature_df)[0]
                confidence = max(probabilities)
                class_probabilities = dict(enumerate(probabilities))
            else:
                prediction = self.model.predict(feature_df)[0]
                confidence = abs(prediction) if isinstance(prediction, (int, float)) else 0.5
                class_probabilities = None
                
        except Exception as e:
            return {
                'error': f"Prediction failed: {e}",
                'features_extracted': len(features),
                'feature_names': list(features.keys())
            }
        
        # Prepare user-friendly results
        results = self._format_user_friendly_results(
            json_data, features, prediction, confidence, class_probabilities, feature_df
        )
            
        return results
    
    def _format_user_friendly_results(self, json_data, features, prediction, confidence, 
                                    class_probabilities, feature_df) -> Dict[str, Any]:
        """Format results in a user-friendly way"""
        
        # Define class labels (customize these based on your model)
        class_labels = {
            0: {"name": "Benign", "description": "Safe and legitimate application", "risk": "LOW"},
            1: {"name": "Suspicious", "description": "Potentially harmful or deceptive app", "risk": "MEDIUM"},
            2: {"name": "Adware", "description": "Contains aggressive advertising", "risk": "MEDIUM"},
            3: {"name": "Malware", "description": "Malicious software detected", "risk": "HIGH"},
            4: {"name": "PUA", "description": "Potentially Unwanted Application", "risk": "MEDIUM"}
        }
        
        manifest = json_data.get('manifest', {})
        package_name = manifest.get('package', 'Unknown')
        
        # Get predicted class info
        predicted_class = class_labels.get(prediction, {
            "name": f"Class {prediction}", 
            "description": "Unknown classification",
            "risk": "UNKNOWN"
        })
        
        # Risk level with color coding
        risk_colors = {
            "LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴", "UNKNOWN": "⚪"
        }
        
        # Security analysis
        security_analysis = self._analyze_security_features(features, manifest)
        
        # Confidence interpretation
        confidence_level = "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"
        confidence_emoji = "🎯" if confidence > 0.8 else "🎲" if confidence > 0.6 else "❓"
        
        results = {
            "📱 APP INFORMATION": {
                "App Name": package_name.split('.')[-1].title(),
                "Package Name": package_name,
                "Version": manifest.get('version_name', 'Unknown'),
                "File Location": json_data.get('apk', 'Unknown').split('\\')[-1] if '\\' in json_data.get('apk', '') else json_data.get('apk', 'Unknown').split('/')[-1]
            },
            
            "🛡️ SECURITY VERDICT": {
                "Classification": f"{risk_colors[predicted_class['risk']]} {predicted_class['name']}",
                "Risk Level": f"{risk_colors[predicted_class['risk']]} {predicted_class['risk']}",
                "Description": predicted_class['description'],
                "Confidence": f"{confidence_emoji} {confidence_level} ({confidence:.1%})"
            },
            
            "📊 DETAILED ANALYSIS": security_analysis,
            
            "🎯 PREDICTION BREAKDOWN": self._format_class_probabilities(class_probabilities, class_labels) if class_probabilities else "Not available",
            
            "⚙️ TECHNICAL SUMMARY": {
                "Features Analyzed": len(feature_df.columns),
                "Model Type": "XGBoost Classifier",
                "Analysis Method": "Static + Manifest Analysis"
            },
            
            "📋 RECOMMENDATIONS": self._generate_recommendations(predicted_class, security_analysis, confidence),
            
            "🏁 FINAL VERDICT": self._generate_final_verdict(predicted_class, confidence, security_analysis)
        }
        
        return results
    
    def _generate_final_verdict(self, predicted_class, confidence, security_analysis) -> str:
        """Generate a final verdict about app genuineness"""
        
        risk_level = predicted_class["risk"]
        
        # Base verdict on risk level and confidence
        if risk_level == "LOW" and confidence > 0.7:
            verdict = "✅ GENUINE - This app appears to be legitimate and safe to use"
        elif risk_level == "LOW" and confidence > 0.5:
            verdict = "✅ LIKELY GENUINE - This app appears legitimate but monitor its behavior"
        elif risk_level == "MEDIUM" and confidence > 0.7:
            verdict = "⚠️ NOT GENUINE - This app shows suspicious characteristics and should be avoided"
        elif risk_level == "MEDIUM":
            verdict = "❓ QUESTIONABLE - Cannot determine if this app is genuine. Exercise extreme caution"
        elif risk_level == "HIGH":
            verdict = "🚫 NOT GENUINE - This app is likely malicious or fake. DO NOT INSTALL"
        else:
            verdict = "❓ UNCERTAIN - Unable to determine app genuineness with current analysis"
        
        # Add specific context for fake apps
        package_name = security_analysis.get("Package Name", "").lower()
        if "fake" in package_name or "simulator" in package_name or "prank" in package_name:
            if risk_level in ["LOW", "MEDIUM"]:
                verdict += "\n💡 NOTE: App name suggests it's intentionally fake (simulator/prank app)"
        
        # Add confidence context
        if confidence < 0.6:
            verdict += f"\n🎲 Confidence is {confidence:.1%} - consider additional security checks"
        
        return verdict
    
    def _analyze_security_features(self, features, manifest) -> Dict[str, Any]:
        """Analyze security-related features in a user-friendly way"""
        
        analysis = {}
        
        # Permission Analysis
        permissions = manifest.get('permissions', [])
        dangerous_count = features.get('num_dangerous_permissions', 0)
        total_permissions = features.get('num_permissions', 0)
        
        if dangerous_count > 0:
            perm_status = f"⚠️ {dangerous_count} dangerous permissions detected"
        elif total_permissions > 10:
            perm_status = f"🟡 {total_permissions} permissions (many requested)"
        elif total_permissions > 5:
            perm_status = f"🟢 {total_permissions} permissions (moderate)"
        else:
            perm_status = f"🟢 {total_permissions} permissions (minimal)"
            
        analysis["Permission Security"] = perm_status
        
        # Specific Permission Flags
        risky_permissions = []
        if features.get('has_internet'): risky_permissions.append("Internet Access")
        if features.get('has_location'): risky_permissions.append("Location Access")
        if features.get('has_sms'): risky_permissions.append("SMS Access")
        if features.get('has_phone'): risky_permissions.append("Phone Access")
        if features.get('has_camera'): risky_permissions.append("Camera Access")
        if features.get('has_contacts'): risky_permissions.append("Contacts Access")
        if features.get('has_storage'): risky_permissions.append("Storage Access")
        if features.get('has_admin'): risky_permissions.append("Device Admin")
        
        if risky_permissions:
            analysis["Sensitive Permissions"] = ", ".join(risky_permissions)
        else:
            analysis["Sensitive Permissions"] = "None detected"
        
        # App Structure Analysis
        total_components = features.get('total_components', 0)
        if total_components > 20:
            structure_status = f"🟡 Complex app ({total_components} components)"
        elif total_components > 10:
            structure_status = f"🟢 Medium complexity ({total_components} components)"
        else:
            structure_status = f"🟢 Simple structure ({total_components} components)"
            
        analysis["App Complexity"] = structure_status
        
        # Suspicious Indicators
        suspicious_score = features.get('component_suspicious_score', 0) + features.get('has_suspicious_keywords', 0)
        if suspicious_score > 0:
            analysis["Suspicious Indicators"] = f"⚠️ {suspicious_score} suspicious patterns found"
        else:
            analysis["Suspicious Indicators"] = "🟢 No suspicious patterns detected"
        
        # SDK Analysis
        min_sdk = features.get('min_sdk', 0)
        target_sdk = features.get('target_sdk', 0)
        if target_sdk < 28:
            analysis["SDK Security"] = f"🟡 Targets older Android (API {target_sdk})"
        elif target_sdk >= 30:
            analysis["SDK Security"] = f"🟢 Modern Android target (API {target_sdk})"
        else:
            analysis["SDK Security"] = f"🟢 Recent Android target (API {target_sdk})"
        
        return analysis
    
    def _format_class_probabilities(self, class_probabilities, class_labels) -> Dict[str, str]:
        """Format class probabilities in a user-friendly way"""
        formatted = {}
        
        # Sort by probability (highest first)
        sorted_probs = sorted(class_probabilities.items(), key=lambda x: x[1], reverse=True)
        
        for class_id, prob in sorted_probs:
            class_info = class_labels.get(int(class_id), {"name": f"Class {class_id}", "risk": "UNKNOWN"})
            risk_emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴", "UNKNOWN": "⚪"}[class_info["risk"]]
            
            if prob > 0.01:  # Only show probabilities > 1%
                formatted[f"{risk_emoji} {class_info['name']}"] = f"{prob:.1%}"
        
        return formatted
    
    def _generate_recommendations(self, predicted_class, security_analysis, confidence) -> List[str]:
        """Generate user-friendly recommendations"""
        recommendations = []
        
        risk_level = predicted_class["risk"]
        
        if risk_level == "HIGH":
            recommendations.extend([
                "🚫 DO NOT INSTALL this application",
                "🛡️ Run a full device scan if already installed",
                "📱 Consider using antivirus software",
                "⚠️ Report this app if found on official stores"
            ])
        elif risk_level == "MEDIUM":
            recommendations.extend([
                "⚠️ Use caution before installing",
                "📋 Review app permissions carefully",
                "🔍 Verify the developer's reputation",
                "📱 Monitor app behavior after installation"
            ])
        elif risk_level == "LOW":
            recommendations.extend([
                "✅ App appears safe for installation",
                "📋 Still review permissions as good practice",
                "📱 Monitor for unusual behavior"
            ])
        else:
            recommendations.append("❓ Unable to determine risk level - proceed with caution")
        
        # Confidence-based recommendations
        if confidence < 0.6:
            recommendations.append("🎲 Low confidence prediction - consider additional analysis")
        
        # Permission-based recommendations
        if "dangerous permissions" in security_analysis.get("Permission Security", "").lower():
            recommendations.append("🔒 Review why this app needs dangerous permissions")
        
        if "Device Admin" in security_analysis.get("Sensitive Permissions", ""):
            recommendations.append("👨‍💼 Be very careful with Device Admin permissions")
        
        return recommendations


def main():
    parser = argparse.ArgumentParser(
        description="🛡️ APK Security Analysis Tool - Test your APK files for potential threats"
    )
    parser.add_argument('--model', required=True, help='Path to model pickle file')
    parser.add_argument('--data', required=True, help='Path to JSON analysis file')
    parser.add_argument('--csv', help='Path to CSV frequency file (optional)')
    parser.add_argument('--output', help='Output file for results (optional)')
    parser.add_argument('--verbose', action='store_true', help='Show detailed technical output')
    parser.add_argument('--json', action='store_true', help='Output in JSON format (for technical users)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        sys.exit(1)
        
    if not os.path.exists(args.data):
        print(f"❌ Data file not found: {args.data}")
        sys.exit(1)
        
    if args.csv and not os.path.exists(args.csv):
        print(f"❌ CSV file not found: {args.csv}")
        sys.exit(1)
    
    try:
        # Show loading progress
        if not args.json:
            print("🔄 Loading AI security model...")
        tester = ModelTester(args.model)
        
        if not args.json:
            print("🔍 Analyzing APK file...")
        results = tester.test_model(args.data, args.csv)
        
        # Handle errors
        if 'error' in results:
            print(f"❌ Analysis failed: {results['error']}")
            sys.exit(1)
        
        # Output results based on format preference
        if args.json:
            # JSON format for technical users
            print(json.dumps(results, indent=2, ensure_ascii=False))
        else:
            # User-friendly format
            print("\n" + "="*60)
            print("🛡️  APK SECURITY ANALYSIS REPORT")
            print("="*60)
            
            for section, content in results.items():
                print(f"\n{section}")
                print("-" * len(section))
                
                if isinstance(content, dict):
                    for key, value in content.items():
                        if isinstance(value, list):
                            print(f"{key}:")
                            for item in value:
                                print(f"  • {item}")
                        else:
                            print(f"{key}: {value}")
                elif isinstance(content, list):
                    for item in content:
                        print(f"• {item}")
                else:
                    print(content)
            
            print("\n" + "="*60)
            print("Analysis completed successfully! 🎉")
            print("="*60)
        
        # Save to file if requested
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            if not args.json:
                print(f"📁 Full report saved to: {args.output}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Analysis cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()