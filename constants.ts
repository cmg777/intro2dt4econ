
import { DataPoint } from './types';

// Generate synthetic data mimicking the image's distribution
export const INITIAL_DATA: DataPoint[] = [
  // High Income, High Education (Blue Zone - Job Found)
  { id: 1, education: 18, income: 80, foundJob: true },
  { id: 2, education: 17, income: 70, foundJob: true },
  { id: 3, education: 19, income: 40, foundJob: true },
  { id: 4, education: 16.5, income: 60, foundJob: true },
  
  // High Income, Low Education (Pink Zone - No Job)
  { id: 5, education: 5, income: 85, foundJob: false },
  { id: 6, education: 8, income: 75, foundJob: false },
  { id: 7, education: 4, income: 65, foundJob: false },
  { id: 8, education: 12, income: 80, foundJob: false },
  { id: 9, education: 10, income: 55, foundJob: false },
  { id: 10, education: 14, income: 70, foundJob: false },
  { id: 11, education: 2, income: 45, foundJob: false },
  { id: 12, education: 7, income: 30, foundJob: false },

  // Low Income, High Education (Green Zone - Job Found)
  { id: 13, education: 14, income: 15, foundJob: true },
  { id: 14, education: 16, income: 10, foundJob: true },
  { id: 15, education: 18, income: 5, foundJob: true },
  { id: 16, education: 12, income: 8, foundJob: true },
  { id: 17, education: 15, income: 12, foundJob: true },
  { id: 18, education: 19, income: 2, foundJob: true },
  { id: 19, education: 13, income: 18, foundJob: true },

  // Low Income, Low Education (Orange Zone - No Job)
  { id: 20, education: 4, income: 15, foundJob: false },
  { id: 21, education: 8, income: 10, foundJob: false },
  { id: 22, education: 2, income: 5, foundJob: false },
  { id: 23, education: 6, income: 12, foundJob: false },
  { id: 24, education: 1, income: 18, foundJob: false },
];

export const COLORS = {
  // Background zones (Pastels)
  zoneBlue: '#e0f2fe',   // Sky 100
  zonePink: '#fee2e2',   // Red 100
  zoneGreen: '#dcfce7',  // Green 100
  zoneOrange: '#ffedd5', // Orange 100

  // Border/Accent colors for zones
  borderBlue: '#7dd3fc',
  borderPink: '#fca5a5',
  borderGreen: '#86efac',
  borderOrange: '#fdba74',

  // Data points
  pointFound: '#15803d', // Green 700
  pointNotFound: '#b91c1c', // Red 700
  
  // UI Elements
  axis: '#94a3b8',
  grid: '#e2e8f0',
  splitLine: '#475569'
};
