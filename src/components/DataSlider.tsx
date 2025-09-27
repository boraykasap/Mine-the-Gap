import { Slider } from '@/components/ui/slider';
import { Card } from '@/components/ui/card';

interface DataSliderProps {
  min: number;
  max: number;
  value: [number, number];
  onChange: (value: [number, number]) => void;
  uniqueValues: (string | number)[];
}

export const DataSlider = ({ min, max, value, onChange, uniqueValues }: DataSliderProps) => {
  const getDisplayValue = (index: number) => {
    return uniqueValues[index] || index;
  };

  return (
    <Card className="p-6 mb-6 bg-gradient-to-r from-background to-secondary/30 border-primary/20">
      <div className="space-y-4">
        <div className="flex justify-between items-center">
          <h3 className="text-lg font-semibold text-foreground">Filter by ID Range</h3>
          <div className="flex items-center gap-4 text-sm text-muted-foreground">
            <span>From: <span className="font-medium text-foreground">{getDisplayValue(value[0])}</span></span>
            <span>To: <span className="font-medium text-foreground">{getDisplayValue(value[1])}</span></span>
          </div>
        </div>
        <div className="px-2">
          <Slider
            value={value}
            onValueChange={onChange}
            min={min}
            max={max}
            step={1}
            className="w-full"
          />
        </div>
        <div className="flex justify-between text-xs text-muted-foreground">
          <span>{getDisplayValue(min)}</span>
          <span>{getDisplayValue(max)}</span>
        </div>
      </div>
    </Card>
  );
};