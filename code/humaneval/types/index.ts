export interface Question {
  id: string;
  query: string;
  llmResponse: string;
  finetunedResponse: string;
}

export interface Rating {
  id: string;
  questionId: string;
  username: string;
  llmRatings: {
    [key: string]: number;
  };
  finetunedRatings: {
    [key: string]: number;
  };
  createdAt: string;
}

export interface EvaluationQuestion {
  id: string;
  name: string;
} 