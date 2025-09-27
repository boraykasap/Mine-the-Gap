import { useState, useMemo } from 'react';
import { CSVUpload } from '@/components/CSVUpload';
import { DataTable } from '@/components/DataTable';
import { DataSlider } from '@/components/DataSlider';
import { toast } from '@/hooks/use-toast';

const Index = () => {
  const [csvData, setCsvData] = useState<string[][]>([]);
  const [filterRange, setFilterRange] = useState<[number, number]>([0, 0]);
  const [columnFilters, setColumnFilters] = useState<Record<number, string[]>>({});

  // Get unique values from first column (excluding header)
  const uniqueFirstColumnValues = useMemo(() => {
    if (csvData.length <= 1) return [];
    return csvData.slice(1).map(row => row[0]).filter((value, index, self) => self.indexOf(value) === index);
  }, [csvData]);

  // Create filtered data based on slider range and column filters
  const filteredData = useMemo(() => {
    if (csvData.length === 0) return [];
    
    const header = csvData[0];
    const dataRows = csvData.slice(1);
    
    // Apply range filter first
    let filteredRows = dataRows.filter((row, index) => {
      return index >= filterRange[0] && index <= filterRange[1];
    });
    
    // Apply column filters
    Object.entries(columnFilters).forEach(([columnIndex, selectedValues]) => {
      if (selectedValues.length > 0) {
        const colIndex = parseInt(columnIndex);
        filteredRows = filteredRows.filter(row => 
          selectedValues.includes(row[colIndex])
        );
      }
    });
    
    return [header, ...filteredRows];
  }, [csvData, filterRange, columnFilters]);

  const handleFileUpload = (data: string[][]) => {
    setCsvData(data);
    setFilterRange([0, Math.max(0, data.length - 2)]);
    setColumnFilters({}); // Clear column filters when new data is uploaded
    toast({
      title: "CSV Uploaded",
      description: `Successfully loaded ${data.length - 1} rows of data`,
    });
  };

  const handleFilterChange = (value: [number, number]) => {
    setFilterRange(value);
  };

  const handleColumnFilterChange = (columnIndex: number, selectedValues: string[]) => {
    setColumnFilters(prev => ({
      ...prev,
      [columnIndex]: selectedValues
    }));
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto py-8 px-4 max-w-7xl">
        {/* Header with Upload */}
        <div className="flex justify-between items-center mb-8">
          <div className="space-y-2">
            <h1 className="text-4xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
              CSV Data Viewer
            </h1>
            <p className="text-muted-foreground text-lg">
              Upload and explore your CSV data with interactive filtering
            </p>
          </div>
          <CSVUpload onFileUpload={handleFileUpload} />
        </div>

        {/* Slider Filter */}
        {csvData.length > 0 && (
          <DataSlider
            min={0}
            max={Math.max(0, csvData.length - 2)}
            value={filterRange}
            onChange={handleFilterChange}
            uniqueValues={uniqueFirstColumnValues}
          />
        )}

        {/* Data Table */}
        <DataTable 
          data={csvData} 
          filteredData={filteredData}
          columnFilters={columnFilters}
          onColumnFilterChange={handleColumnFilterChange}
        />
      </div>
    </div>
  );
};

export default Index;
