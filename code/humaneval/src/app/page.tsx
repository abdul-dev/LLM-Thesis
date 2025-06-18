'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Button } from '@/components/ui/button';
import { sampleQuestions } from '@/lib/sample-data';
import { evaluationQuestions } from '@/lib/constants';
import { toast } from 'sonner';
import { ChevronLeft, ChevronRight, BarChart2, Trash2 } from 'lucide-react';
import Link from 'next/link';

export default function Home() {
  const [username, setUsername] = useState('');
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [ratings, setRatings] = useState<{
    [sampleId: string]: {
      [evalId: string]: {
        llm: number;
        finetuned: number;
      }
    }
  }>({});

  // Load ratings from localStorage on component mount
  useEffect(() => {
    const savedRatings = localStorage.getItem('questionRatings');
    if (savedRatings) {
      setRatings(JSON.parse(savedRatings));
    }
  }, []);

  // Save ratings to localStorage whenever they change
  useEffect(() => {
    localStorage.setItem('questionRatings', JSON.stringify(ratings));
  }, [ratings]);

  const calculateOverallScore = (questionId: string, type: 'llm' | 'finetuned') => {
    const scores = evaluationQuestions.map(q => 
      ratings[questionId]?.[q.id]?.[type] || 0
    );
    const sum = scores.reduce((a, b) => a + b, 0);
    return Math.round((sum / scores.length) * 10) / 10;
  };

  const handleRatingChange = (
    sampleId: string,
    evalId: string,
    type: 'llm' | 'finetuned',
    value: number[]
  ) => {
    setRatings(prev => ({
      ...prev,
      [sampleId]: {
        ...prev[sampleId],
        [evalId]: {
          ...prev[sampleId]?.[evalId],
          [type]: value[0]
        }
      }
    }));
  };

  const handleClearData = () => {
    if (window.confirm('Are you sure you want to clear all ratings?')) {
      localStorage.removeItem('questionRatings');
      setRatings({});
      toast.success('All ratings have been cleared');
    }
  };

  const handleSubmit = async () => {
    if (!username) {
      toast.error('Please enter your username');
      return;
    }

    // Check if all questions are rated
    const allQuestionsRated = sampleQuestions.every(sampleQuestion =>
      evaluationQuestions.every(evalQuestion =>
        ratings[sampleQuestion.id]?.[evalQuestion.id]?.llm !== undefined &&
        ratings[sampleQuestion.id]?.[evalQuestion.id]?.finetuned !== undefined
      )
    );

    if (!allQuestionsRated) {
      toast.error('Please rate all questions for both responses');
      return;
    }

    toast.success('Ratings submitted successfully!');
  };

  const currentQuestion = sampleQuestions[currentQuestionIndex];

  return (
    <main className="min-h-screen p-6">
      <div className="max-w-[2000px] mx-auto">
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-3xl font-bold">Human Evalution System</h1>
          <div className="flex gap-4">
            <Link href="/dashboard">
              <Button variant="outline" className="flex items-center gap-2">
                <BarChart2 className="h-4 w-4" />
                View Dashboard
              </Button>
            </Link>
            <Button 
              variant="outline" 
              className="flex items-center gap-2 text-red-600 hover:text-red-700"
              onClick={handleClearData}
            >
              <Trash2 className="h-4 w-4" />
              Clear Data
            </Button>
          </div>
        </div>
        
        <div className="mb-6">
          <Label htmlFor="username">Username</Label>
          <Input
            id="username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            placeholder="Enter your username"
            className="mt-2 max-w-md"
          />
        </div>

        <div className="grid grid-cols-12 gap-8">
          {/* Main Content - Current Sample Question */}
          <div className="col-span-9">
            <Card>
              <div className="p-8">
                <div className="mb-8">
                  <div className="flex justify-between items-center mb-4">
                    <h2 className="text-2xl font-semibold">Question {currentQuestion.id}</h2>
                    <div className="text-sm text-muted-foreground">
                      {currentQuestionIndex + 1} of {sampleQuestions.length}
                    </div>
                  </div>
                  <p className="text-base text-muted-foreground">{currentQuestion.query}</p>
                </div>

                <div className="grid grid-cols-2 gap-8">
                  <div>
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="font-semibold text-lg">LLM Response</h3>
                      <span className="text-xs px-2 py-1 bg-blue-100 text-blue-800 rounded-full">AI Generated</span>
                    </div>
                    <div className="bg-muted/50 p-6 rounded-lg border">
                      <p className="text-base leading-relaxed">{currentQuestion.llmResponse}</p>
                    </div>
                  </div>

                  <div>
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="font-semibold text-lg">Finetuned Response</h3>
                      <span className="text-xs px-2 py-1 bg-green-100 text-green-800 rounded-full">Optimized</span>
                    </div>
                    <div className="bg-muted/50 p-6 rounded-lg border">
                      <p className="text-base leading-relaxed">{currentQuestion.finetunedResponse}</p>
                    </div>
                  </div>
                </div>
              </div>
            </Card>
          </div>

          {/* Right Side - Evaluation Criteria Card */}
          <div className="col-span-3">
            <Card className="sticky top-6">
              <div className="p-4">
                <h3 className="font-semibold mb-3 text-sm uppercase tracking-wider text-muted-foreground">
                  Evaluation Criteria
                </h3>
                <div className="space-y-3">
                  {evaluationQuestions.map((evalQuestion) => (
                    <div key={evalQuestion.id} className="space-y-1.5">
                      <div className="flex items-start justify-between gap-2">
                        <div>
                          <h4 className="font-medium text-sm">{evalQuestion.name}</h4>
                          <p className="text-xs text-muted-foreground">{evalQuestion.description}</p>
                        </div>
                        <div className="flex gap-2 text-xs text-muted-foreground shrink-0">
                          <span>L: {ratings[currentQuestion.id]?.[evalQuestion.id]?.llm || 0}</span>
                          <span>F: {ratings[currentQuestion.id]?.[evalQuestion.id]?.finetuned || 0}</span>
                        </div>
                      </div>
                      <div className="grid grid-cols-2 gap-2">
                        <div>
                          <Label className="text-xs text-muted-foreground">LLM</Label>
                          <Slider
                            defaultValue={[0]}
                            max={10}
                            step={1}
                            value={[ratings[currentQuestion.id]?.[evalQuestion.id]?.llm || 0]}
                            onValueChange={(value) => 
                              handleRatingChange(currentQuestion.id, evalQuestion.id, 'llm', value)
                            }
                            className="mt-1"
                          />
                        </div>
                        <div>
                          <Label className="text-xs text-muted-foreground">Finetuned</Label>
                          <Slider
                            defaultValue={[0]}
                            max={10}
                            step={1}
                            value={[ratings[currentQuestion.id]?.[evalQuestion.id]?.finetuned || 0]}
                            onValueChange={(value) => 
                              handleRatingChange(currentQuestion.id, evalQuestion.id, 'finetuned', value)
                            }
                            className="mt-1"
                          />
                        </div>
                      </div>
                    </div>
                  ))}

                  {/* Overall Score */}
                  <div className="pt-3 border-t">
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <div>
                          <h4 className="font-medium text-sm">Overall Score</h4>
                          <p className="text-xs text-muted-foreground">Weighted average</p>
                        </div>
                        <div className="flex gap-4">
                          <div className="text-right">
                            <div className="text-xs text-muted-foreground">LLM</div>
                            <div className="text-lg font-semibold">
                              {calculateOverallScore(currentQuestion.id, 'llm')}/10
                            </div>
                          </div>
                          <div className="text-right">
                            <div className="text-xs text-muted-foreground">Finetuned</div>
                            <div className="text-lg font-semibold">
                              {calculateOverallScore(currentQuestion.id, 'finetuned')}/10
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </Card>
          </div>
        </div>

        {/* Navigation Controls */}
        <div className="mt-8 flex justify-between items-center">
          <Button
            variant="outline"
            onClick={() => setCurrentQuestionIndex(prev => Math.max(0, prev - 1))}
            disabled={currentQuestionIndex === 0}
            className="flex items-center gap-2"
          >
            <ChevronLeft className="h-4 w-4" />
            Previous Question
          </Button>

          <div className="text-sm text-muted-foreground">
            Question {currentQuestionIndex + 1} of {sampleQuestions.length}
          </div>

          <Button
            variant="outline"
            onClick={() => setCurrentQuestionIndex(prev => Math.min(sampleQuestions.length - 1, prev + 1))}
            disabled={currentQuestionIndex === sampleQuestions.length - 1}
            className="flex items-center gap-2"
          >
            Next Question
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>

        <div className="mt-8">
          <Button onClick={handleSubmit} className="w-full max-w-md mx-auto">
            Submit Ratings
          </Button>
        </div>
      </div>
    </main>
  );
} 