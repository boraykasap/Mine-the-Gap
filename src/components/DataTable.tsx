import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { Check, ChevronDown, Filter, X } from 'lucide-react';

interface DataTableProps {
  data: string[][];
  filteredData: string[][];
  columnFilters: Record<number, string[]>;
  onColumnFilterChange: (columnIndex: number, selectedValues: string[]) => void;
}

export const DataTable = ({ data, filteredData, columnFilters, onColumnFilterChange }: DataTableProps) => {
  if (data.length === 0) {
    return (
      <Card className="upload-area">
        <div className="space-y-4">
          <div className="mx-auto w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center">
            <svg className="w-8 h-8 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          </div>
          <div>
            <h3 className="text-lg font-semibold text-foreground">No Data</h3>
            <p className="text-muted-foreground">Upload a CSV file to view your data</p>
          </div>
        </div>
      </Card>
    );
  }

  const headers = data[0];
  const rows = filteredData.slice(1);

  // Get unique values for each column
  const getUniqueValuesForColumn = (columnIndex: number) => {
    if (data.length <= 1) return [];
    const values = data.slice(1).map(row => row[columnIndex]).filter(Boolean);
    return [...new Set(values)].sort();
  };

  const handleFilterToggle = (columnIndex: number, value: string) => {
    const currentFilters = columnFilters[columnIndex] || [];
    const newFilters = currentFilters.includes(value)
      ? currentFilters.filter(v => v !== value)
      : [...currentFilters, value];
    onColumnFilterChange(columnIndex, newFilters);
  };

  const clearColumnFilter = (columnIndex: number) => {
    onColumnFilterChange(columnIndex, []);
  };

  return (
    <Card className="table-container">
      <Table>
        <TableHeader>
          <TableRow className="table-header border-b-2">
            {headers.map((header, index) => {
              const uniqueValues = getUniqueValuesForColumn(index);
              const activeFilters = columnFilters[index] || [];
              const hasActiveFilter = activeFilters.length > 0;
              
              return (
                <TableHead key={index} className="font-semibold text-secondary-foreground py-4">
                  <div className="flex items-center justify-between gap-2">
                    <span>{header}</span>
                    {uniqueValues.length > 1 && (
                      <Popover>
                        <PopoverTrigger asChild>
                          <Button
                            variant="ghost"
                            size="sm"
                            className={`h-6 w-6 p-0 ${hasActiveFilter ? 'text-primary' : 'text-muted-foreground hover:text-foreground'}`}
                          >
                            <Filter className="h-3 w-3" />
                          </Button>
                        </PopoverTrigger>
                        <PopoverContent className="w-56 p-2" align="start">
                          <div className="space-y-2">
                            <div className="flex items-center justify-between">
                              <span className="text-sm font-medium">Filter {header}</span>
                              {hasActiveFilter && (
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  onClick={() => clearColumnFilter(index)}
                                  className="h-6 w-6 p-0 text-muted-foreground hover:text-foreground"
                                >
                                  <X className="h-3 w-3" />
                                </Button>
                              )}
                            </div>
                            <div className="max-h-48 overflow-y-auto space-y-1">
                              {uniqueValues.map((value) => {
                                const isSelected = activeFilters.includes(value);
                                return (
                                  <div
                                    key={value}
                                    className="flex items-center space-x-2 p-1 rounded hover:bg-muted cursor-pointer"
                                    onClick={() => handleFilterToggle(index, value)}
                                  >
                                    <div className={`w-4 h-4 border rounded flex items-center justify-center ${isSelected ? 'bg-primary border-primary' : 'border-muted-foreground'}`}>
                                      {isSelected && <Check className="h-2.5 w-2.5 text-primary-foreground" />}
                                    </div>
                                    <span className="text-sm truncate flex-1">{value}</span>
                                  </div>
                                );
                              })}
                            </div>
                            {hasActiveFilter && (
                              <div className="text-xs text-muted-foreground pt-1 border-t">
                                {activeFilters.length} of {uniqueValues.length} selected
                              </div>
                            )}
                          </div>
                        </PopoverContent>
                      </Popover>
                    )}
                  </div>
                </TableHead>
              );
            })}
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((row, rowIndex) => (
            <TableRow key={rowIndex} className="table-row">
              {row.map((cell, cellIndex) => (
                <TableCell key={cellIndex} className="py-3">
                  {cell}
                </TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
      {rows.length === 0 && (
        <div className="p-8 text-center text-muted-foreground">
          No data matches the current filter range
        </div>
      )}
    </Card>
  );
};