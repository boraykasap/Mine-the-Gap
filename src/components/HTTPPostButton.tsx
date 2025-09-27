import { Upload } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { useRef } from 'react';

interface CSVUploadProps {
  onFileUpload: (data: string[][]) => void;
}

export const HTTPPostButton = ({ onFileUpload }: CSVUploadProps) => {
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

  const handleHTTPPostRequest = () => {
    
    console.log(document.getElementById("textfield").textContent);
    
    fetch("http://localhost:8000/items", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      content: document.getElementById("TextInput01").value
    })
  })
  .then(response => {
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
  })
  .then(data => {
    console.log("Success:", data);
    document.getElementById("textfield").textContent=JSON.stringify(data);
  })
  .catch(error => {
    console.error("Error:", error);
  });


  };

  return (
    <div className="flex items-center gap-4">
      <Button 
        onClick={handleHTTPPostRequest}
        variant="default"
        className="bg-gradient-to-r from-primary to-accent hover:from-primary/90 hover:to-accent/90 shadow-lg transition-all duration-300 hover:shadow-xl hover:-translate-y-0.5"
      >
        HTTPPostButton
      </Button>
    </div>
  );
};