import { Upload } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { useRef } from 'react';

interface CSVUploadProps {
  onFileUpload: (data: string[][]) => void;
}

export const CSVUpload = ({ onFileUpload }: CSVUploadProps) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type === 'text/csv') {
      const reader = new FileReader();
      reader.onload = (e) => {
        const text = e.target?.result as string;
        const rows = text.split('\n').map(row => 
          row.split(',').map(cell => cell.trim().replace(/^"|"$/g, ''))
        ).filter(row => row.some(cell => cell.length > 0));
        onFileUpload(rows);
      };
      reader.readAsText(file);
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="flex items-center gap-4">
      <Button 
        onClick={handleUploadClick}
        variant="default"
        className="bg-gradient-to-r from-primary to-accent hover:from-primary/90 hover:to-accent/90 shadow-lg transition-all duration-300 hover:shadow-xl hover:-translate-y-0.5"
      >
        <Upload className="mr-2 h-4 w-4" />
        Upload CSV
      </Button>
      <Input
        ref={fileInputRef}
        type="file"
        accept=".csv"
        onChange={handleFileChange}
        className="hidden"
      />
    </div>
  );
};