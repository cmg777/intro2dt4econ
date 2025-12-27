
export interface DataPoint {
  id: number;
  education: number;
  income: number;
  foundJob: boolean;
}

export interface Split {
  feature: 'education' | 'income';
  threshold: number;
  direction: 'left' | 'right';
}

export interface TreeNode {
  id: string;
  name: string;
  condition?: string;
  prediction?: 'Found a job' | "Didn't find a job";
  children?: TreeNode[];
  color?: string;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}
