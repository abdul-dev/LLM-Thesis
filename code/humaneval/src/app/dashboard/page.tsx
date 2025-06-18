'use client';

import { useEffect, useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { sampleQuestions } from '@/lib/sample-data';
import { evaluationQuestions } from '@/lib/constants';
import Link from 'next/link';
import { ArrowLeft, TrendingUp, BarChart2, Sparkles } from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';

export default function Dashboard() {
  const [ratings, setRatings] = useState<{
    [sampleId: string]: {
      [evalId: string]: {
        llm: number;
        finetuned: number;
      }
    }
  }>({});

  useEffect(() => {
    const savedRatings = localStorage.getItem('questionRatings');
    if (savedRatings) {
      setRatings(JSON.parse(savedRatings));
    }
  }, []);

  const calculateAverageScore = (questionId: string, type: 'llm' | 'finetuned') => {
    const scores = evaluationQuestions.map(q => 
      ratings[questionId]?.[q.id]?.[type] || 0
    );
    const sum = scores.reduce((a, b) => a + b, 0);
    return Math.round((sum / scores.length) * 10) / 10;
  };

  const getChartData = (questionId: string) => {
    return evaluationQuestions.map(evalQuestion => ({
      name: evalQuestion.name,
      LLM: ratings[questionId]?.[evalQuestion.id]?.llm || 0,
      Finetuned: ratings[questionId]?.[evalQuestion.id]?.finetuned || 0,
    }));
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100/50">
      <div className="max-w-[2000px] mx-auto p-4">
        {/* Header Section */}
        <div className="flex items-center justify-between mb-6 bg-white/80 backdrop-blur-sm rounded-lg p-4 shadow-sm">
          <div className="flex items-center gap-4">
            <Link href="/">
              <Button variant="ghost" size="sm" className="flex items-center gap-2 hover:bg-gray-100">
                <ArrowLeft className="h-4 w-4" />
                Back
              </Button>
            </Link>
            <div className="flex items-center gap-2">
              <BarChart2 className="h-5 w-5 text-blue-600" />
              <h1 className="text-xl font-bold bg-gradient-to-r from-blue-600 to-blue-800 bg-clip-text text-transparent">
                Evaluation Dashboard
              </h1>
            </div>
          </div>
          <div className="flex items-center gap-2 px-3 py-1.5 bg-blue-50 rounded-full">
            <Sparkles className="h-4 w-4 text-blue-600" />
            <span className="text-sm font-medium text-blue-700">Performance Analysis</span>
          </div>
        </div>

        {/* Grid Layout */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {sampleQuestions.map((question) => (
            <Card key={question.id} className="overflow-hidden border-none shadow-lg hover:shadow-xl transition-shadow duration-200">
              {/* Card Header */}
              <div className="p-4 bg-gradient-to-r from-gray-50 to-white border-b">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="px-2 py-0.5 text-xs font-medium bg-blue-100 text-blue-700 rounded-full">
                        Q{question.id}
                      </span>
                      <span className="text-xs text-muted-foreground">
                        {evaluationQuestions.length} criteria
                      </span>
                    </div>
                    <p className="text-sm text-gray-600 line-clamp-2">{question.query}</p>
                  </div>
                  <div className="flex gap-4 shrink-0">
                    <div className="text-right">
                      <div className="text-xs font-medium text-gray-500">LLM</div>
                      <div className="text-lg font-bold text-blue-600">
                        {calculateAverageScore(question.id, 'llm')}
                        <span className="text-xs font-normal text-gray-400">/10</span>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-xs font-medium text-gray-500">Finetuned</div>
                      <div className="text-lg font-bold text-green-600">
                        {calculateAverageScore(question.id, 'finetuned')}
                        <span className="text-xs font-normal text-gray-400">/10</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Chart Section */}
              <div className="h-[220px] p-4 bg-white">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={getChartData(question.id)}
                    margin={{
                      top: 10,
                      right: 20,
                      left: 0,
                      bottom: 0,
                    }}
                  >
                    <defs>
                      <linearGradient id="llmGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#3b82f6" stopOpacity={1.0}/>
                      </linearGradient>
                      <linearGradient id="finetunedGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#22c55e" stopOpacity={1.0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" vertical={false} />
                    <XAxis 
                      dataKey="name" 
                      tick={{ fontSize: 11, fill: '#6b7280' }}
                      interval={0}
                      angle={-45}
                      textAnchor="end"
                      height={60}
                    />
                    <YAxis 
                      domain={[0, 10]} 
                      tick={{ fontSize: 11, fill: '#6b7280' }}
                      width={30}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: 'white',
                        border: '1px solid #e5e7eb',
                        borderRadius: '8px',
                        boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
                        fontSize: '12px'
                      }}
                      cursor={{ fill: 'rgba(0, 0, 0, 0.05)' }}
                    />
                    <Legend 
                      wrapperStyle={{ 
                        paddingTop: '8px',
                        fontSize: '11px'
                      }}
                      iconType="circle"
                    />
                    <Bar 
                      dataKey="LLM" 
                      fill="url(#llmGradient)" 
                      radius={[4, 4, 0, 0]}
                      maxBarSize={30}
                    />
                    <Bar 
                      dataKey="Finetuned" 
                      fill="url(#finetunedGradient)" 
                      radius={[4, 4, 0, 0]}
                      maxBarSize={30}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Card>
          ))}
        </div>
      </div>
    </main>
  );
} 