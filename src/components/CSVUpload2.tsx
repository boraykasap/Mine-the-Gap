import { Upload } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { useRef } from 'react';

interface CSVUploadProps {
  onFileUpload: (data: string[][]) => void;
}

export const CSVUpload2 = ({ onFileUpload }: CSVUploadProps) => {
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

  const handleUploadClick2 = () => {
    
    fetch('http://localhost:8000/')
    .then(response => {
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return console.log(response.json()); // or response.text() for plain text
    })
    .then(data => {
      console.log(data);
    })
    .catch(error => {
      console.error('Error:', error);
    });

    //document.getElementById("textfield").textContent="hi";
    //window.location.href = "http://localhost:8000/";

  };

  return (
    <div className="flex items-center gap-4">
      <Button 
        onClick={handleUploadClick2}
        variant="default"
        className="bg-gradient-to-r from-primary to-accent hover:from-primary/90 hover:to-accent/90 shadow-lg transition-all duration-300 hover:shadow-xl hover:-translate-y-0.5"
      >
        Upload CSV2
      </Button>
    </div>
  );
};