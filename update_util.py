# Utility functions for updating Google Sheets
import gspread
import pandas as pd
import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Callable
import logging
from functools import wraps
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def retry_on_quota_error(max_retries=3, delay=1):
    """Decorator to retry operations on quota errors"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if "quota" in str(e).lower() and attempt < max_retries - 1:
                        time.sleep(delay * (2 ** attempt))  # Exponential backoff
                        continue
                    raise e
            return None
        return wrapper
    return decorator

class AdvancedSheetUpdater:
    """Advanced Google Sheets updater with batch operations, validation, and RAG integration"""
    
    def __init__(self, worksheet):
        self.worksheet = worksheet
        self.cache = {}
        self.cache_timestamp = None
        self.cache_ttl = 300  # 5 minutes
        
    def _get_cached_data(self, force_refresh=False):
        """Get cached sheet data or refresh if needed"""
        current_time = datetime.now()
        
        if (force_refresh or 
            self.cache_timestamp is None or 
            (current_time - self.cache_timestamp).seconds > self.cache_ttl):
            
            try:
                all_values = self.worksheet.get_all_values()
                if all_values:
                    self.cache = {
                        'headers': all_values[0],
                        'data': all_values[1:],
                        'all_values': all_values
                    }
                    self.cache_timestamp = current_time
                    logger.info(f"Cache refreshed with {len(all_values)} rows")
            except Exception as e:
                logger.error(f"Error refreshing cache: {e}")
                return None
                
        return self.cache
    
    def validate_data_type(self, value: Any, expected_type: str) -> tuple[bool, Any]:
        """Validate and convert data types"""
        try:
            if expected_type.lower() == 'int':
                return True, int(float(str(value)))
            elif expected_type.lower() == 'float':
                return True, float(str(value))
            elif expected_type.lower() == 'bool':
                if str(value).lower() in ['true', '1', 'yes', 'on']:
                    return True, True
                elif str(value).lower() in ['false', '0', 'no', 'off']:
                    return True, False
                else:
                    return False, value
            elif expected_type.lower() == 'date':
                # Try to parse various date formats
                for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S']:
                    try:
                        parsed_date = datetime.strptime(str(value), fmt)
                        return True, parsed_date.strftime('%Y-%m-%d')
                    except ValueError:
                        continue
                return False, value
            else:  # string
                return True, str(value)
        except Exception:
            return False, value
    
    @retry_on_quota_error()
    def batch_update_cells(self, updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform batch updates for efficiency
        updates: [{'row': int, 'col': int, 'value': any}, ...]
        """
        try:
            if not updates:
                return {'success': False, 'message': 'No updates provided'}
            
            # Group updates by row for efficiency
            row_updates = {}
            for update in updates:
                row = update['row']
                if row not in row_updates:
                    row_updates[row] = {}
                row_updates[row][update['col']] = update['value']
            
            # Prepare batch update data
            batch_data = []
            for row, cols in row_updates.items():
                for col, value in cols.items():
                    batch_data.append({
                        'range': f'{chr(64 + col)}{row}',  # Convert to A1 notation
                        'values': [[value]]
                    })
            
            # Execute batch update
            self.worksheet.batch_update(batch_data)
            self._get_cached_data(force_refresh=True)  # Refresh cache
            
            return {
                'success': True,
                'message': f'Successfully updated {len(updates)} cells',
                'updated_count': len(updates)
            }
            
        except Exception as e:
            logger.error(f"Batch update error: {e}")
            return {'success': False, 'message': f'Batch update failed: {e}'}
    
    def smart_search(self, query: str, columns: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Advanced search with fuzzy matching and multiple criteria"""
        cache_data = self._get_cached_data()
        if not cache_data:
            return []
        
        headers = cache_data['headers']
        data = cache_data['data']
        
        # If no specific columns, search all
        if columns is None:
            columns = headers
        
        # Parse query for advanced search patterns
        results = []
        query_lower = query.lower()
        
        for i, row in enumerate(data):
            row_dict = dict(zip(headers, row))
            row_dict['_row_index'] = i + 2  # +2 for 1-based indexing and header
            
            # Check if row matches query
            match_score = 0
            for col in columns:
                if col in row_dict:
                    cell_value = str(row_dict[col]).lower()
                    
                    # Exact match (highest score)
                    if query_lower == cell_value:
                        match_score += 10
                    # Contains match
                    elif query_lower in cell_value:
                        match_score += 5
                    # Fuzzy match (simple)
                    elif any(word in cell_value for word in query_lower.split()):
                        match_score += 2
            
            if match_score > 0:
                row_dict['_match_score'] = match_score
                results.append(row_dict)
        
        # Sort by match score
        results.sort(key=lambda x: x['_match_score'], reverse=True)
        return results
    
    def conditional_update(self, condition_func: Callable, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update rows that match a condition function"""
        cache_data = self._get_cached_data()
        if not cache_data:
            return {'success': False, 'message': 'No data available'}
        
        headers = cache_data['headers']
        data = cache_data['data']
        
        batch_updates = []
        updated_rows = []
        
        try:
            for i, row in enumerate(data):
                row_dict = dict(zip(headers, row))
                
                # Check if row matches condition
                if condition_func(row_dict):
                    row_index = i + 2  # +2 for 1-based indexing and header
                    
                    # Prepare updates for this row
                    for column, new_value in updates.items():
                        if column in headers:
                            col_index = headers.index(column) + 1
                            batch_updates.append({
                                'row': row_index,
                                'col': col_index,
                                'value': new_value
                            })
                    
                    updated_rows.append(row_index)
            
            # Execute batch update
            if batch_updates:
                result = self.batch_update_cells(batch_updates)
                result['updated_rows'] = updated_rows
                return result
            else:
                return {'success': True, 'message': 'No rows matched the condition', 'updated_rows': []}
                
        except Exception as e:
            logger.error(f"Conditional update error: {e}")
            return {'success': False, 'message': f'Conditional update failed: {e}'}
    
    def bulk_insert_rows(self, data_rows: List[Dict[str, Any]], validate_schema=True) -> Dict[str, Any]:
        """Insert multiple rows with optional schema validation"""
        cache_data = self._get_cached_data()
        if not cache_data:
            return {'success': False, 'message': 'No data available'}
        
        headers = cache_data['headers']
        
        try:
            # Validate data if requested
            if validate_schema:
                for i, row_data in enumerate(data_rows):
                    for col in row_data.keys():
                        if col not in headers:
                            return {
                                'success': False, 
                                'message': f'Invalid column "{col}" in row {i+1}. Valid columns: {headers}'
                            }
            
            # Prepare rows for insertion
            rows_to_insert = []
            for row_data in data_rows:
                row = []
                for header in headers:
                    row.append(row_data.get(header, ''))
                rows_to_insert.append(row)
            
            # Insert rows
            if rows_to_insert:
                self.worksheet.append_rows(rows_to_insert)
                self._get_cached_data(force_refresh=True)  # Refresh cache
                
                return {
                    'success': True,
                    'message': f'Successfully inserted {len(rows_to_insert)} rows',
                    'inserted_count': len(rows_to_insert)
                }
            else:
                return {'success': True, 'message': 'No rows to insert', 'inserted_count': 0}
                
        except Exception as e:
            logger.error(f"Bulk insert error: {e}")
            return {'success': False, 'message': f'Bulk insert failed: {e}'}
    
    def create_backup(self) -> Dict[str, Any]:
        """Create a backup of current sheet data"""
        try:
            cache_data = self._get_cached_data(force_refresh=True)
            if not cache_data:
                return {'success': False, 'message': 'No data to backup'}
            
            backup_data = {
                'timestamp': datetime.now().isoformat(),
                'headers': cache_data['headers'],
                'data': cache_data['data'],
                'total_rows': len(cache_data['data']) + 1  # +1 for header
            }
            
            return {
                'success': True,
                'message': f'Backup created with {backup_data["total_rows"]} rows',
                'backup_data': backup_data
            }
            
        except Exception as e:
            logger.error(f"Backup creation error: {e}")
            return {'success': False, 'message': f'Backup failed: {e}'}
    
    def analyze_data_quality(self) -> Dict[str, Any]:
        """Analyze data quality and provide insights"""
        cache_data = self._get_cached_data()
        if not cache_data:
            return {'success': False, 'message': 'No data available'}
        
        headers = cache_data['headers']
        data = cache_data['data']
        
        analysis = {
            'total_rows': len(data),
            'total_columns': len(headers),
            'column_analysis': {},
            'data_quality_score': 0,
            'issues': []
        }
        
        try:
            for col_idx, header in enumerate(headers):
                col_data = [row[col_idx] if col_idx < len(row) else '' for row in data]
                
                # Analyze column
                non_empty = [val for val in col_data if val.strip()]
                empty_count = len(col_data) - len(non_empty)
                
                # Detect data type
                numeric_count = sum(1 for val in non_empty if self._is_numeric(val))
                date_count = sum(1 for val in non_empty if self._is_date(val))
                
                analysis['column_analysis'][header] = {
                    'total_values': len(col_data),
                    'non_empty_values': len(non_empty),
                    'empty_values': empty_count,
                    'completeness_rate': len(non_empty) / len(col_data) if col_data else 0,
                    'numeric_values': numeric_count,
                    'date_values': date_count,
                    'unique_values': len(set(non_empty)),
                    'most_common': max(set(non_empty), key=non_empty.count) if non_empty else None
                }
                
                # Check for quality issues
                if empty_count > len(col_data) * 0.3:  # More than 30% empty
                    analysis['issues'].append(f'Column "{header}" has high empty rate ({empty_count}/{len(col_data)})')
            
            # Calculate overall quality score
            completeness_scores = [col['completeness_rate'] for col in analysis['column_analysis'].values()]
            analysis['data_quality_score'] = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0
            
            return {'success': True, 'analysis': analysis}
            
        except Exception as e:
            logger.error(f"Data quality analysis error: {e}")
            return {'success': False, 'message': f'Analysis failed: {e}'}
    
    def _is_numeric(self, value: str) -> bool:
        """Check if value is numeric"""
        try:
            float(value)
            return True
        except ValueError:
            return False
    
    def _is_date(self, value: str) -> bool:
        """Check if value is a date"""
        for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S']:
            try:
                datetime.strptime(value, fmt)
                return True
            except ValueError:
                continue
        return False

# Legacy function for backward compatibility
def update_row(worksheet, row_id, id_column, column_to_update, new_value):
    """Update a specific cell in the Google Sheet (legacy function)"""
    try:
        # Get all values to find the correct row
        all_values = worksheet.get_all_values()
        if not all_values:
            return "❌ Sheet is empty"
        
        headers = all_values[0]
        
        # Find column indices
        try:
            id_col_idx = headers.index(id_column) + 1  # gspread uses 1-based indexing
            update_col_idx = headers.index(column_to_update) + 1
        except ValueError as e:
            return f"❌ Column not found: {e}"
        
        # Find the row with matching ID
        target_row = None
        for i, row in enumerate(all_values[1:], start=2):  # Start from row 2 (skip header)
            if str(row[id_col_idx - 1]) == str(row_id):
                target_row = i
                break
        
        if target_row is None:
            return f"❌ No row found with {id_column} = {row_id}"
        
        # Update the cell
        worksheet.update_cell(target_row, update_col_idx, new_value)
        return f"✅ Successfully updated {column_to_update} to '{new_value}' for {id_column} = {row_id}"
        
    except Exception as e:
        return f"❌ Error updating sheet: {e}"

def create_advanced_updater(worksheet) -> AdvancedSheetUpdater:
    """Factory function to create an advanced sheet updater"""
    return AdvancedSheetUpdater(worksheet)

def smart_update_with_ai(worksheet, query: str, ai_function: Callable) -> Dict[str, Any]:
    """Use AI to intelligently update sheet based on natural language query"""
    updater = AdvancedSheetUpdater(worksheet)
    
    try:
        # Search for relevant rows
        search_results = updater.smart_search(query)
        
        if not search_results:
            return {'success': False, 'message': 'No matching rows found for the query'}
        
        # Use AI to determine what updates to make
        context = {
            'query': query,
            'matching_rows': search_results[:5],  # Limit to top 5 matches
            'headers': updater._get_cached_data()['headers']
        }
        
        ai_response = ai_function(context)
        
        return {
            'success': True,
            'message': 'AI-powered update completed',
            'ai_response': ai_response,
            'matching_rows_count': len(search_results)
        }
        
    except Exception as e:
        logger.error(f"AI update error: {e}")
        return {'success': False, 'message': f'AI update failed: {e}'}

def generate_update_report(worksheet) -> Dict[str, Any]:
    """Generate a comprehensive report of recent updates and data quality"""
    updater = AdvancedSheetUpdater(worksheet)
    
    try:
        # Get data quality analysis
        quality_analysis = updater.analyze_data_quality()
        
        # Create backup for comparison
        backup_result = updater.create_backup()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_quality': quality_analysis.get('analysis', {}),
            'backup_created': backup_result['success'],
            'recommendations': []
        }
        
        # Add recommendations based on analysis
        if quality_analysis['success']:
            analysis = quality_analysis['analysis']
            
            if analysis['data_quality_score'] < 0.8:
                report['recommendations'].append('Consider cleaning data - quality score is below 80%')
            
            for col, col_analysis in analysis['column_analysis'].items():
                if col_analysis['completeness_rate'] < 0.7:
                    report['recommendations'].append(f'Column "{col}" has low completeness ({col_analysis["completeness_rate"]:.1%})')
        
        return {'success': True, 'report': report}
        
    except Exception as e:
        logger.error(f"Report generation error: {e}")
        return {'success': False, 'message': f'Report generation failed: {e}'}
