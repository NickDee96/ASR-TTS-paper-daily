import type { PaperPreview } from '../types/paper';

export const archiveSummary = {
  uniquePapers: 11_678,
  topicAssignments: 11_990,
  updatedAt: '2026-07-23T12:00:00Z',
};

export const topicCounts = {
  ASR: 939,
  TTS: 721,
  'Machine Translation': 1_051,
  'Small Language Models': 5_550,
  'Data Augmentation': 1_406,
  'Synthetic Generation': 2_323,
} as const;

export const previewPapers: PaperPreview[] = [
  {
    id: '2607.18064',
    title: 'Autoresearch with Coding Agents: Generalizers and Metric-Maximizers on Quran Recitation Data',
    abstract: 'An empirical study of coding agents that autonomously search model and data choices for speech-recitation assessment, with attention to generalization beyond a held-out metric.',
    authors: ['Nursultan Askarbekuly', 'Amina Rahman', 'Omar Farooq'],
    published: '2026-07-20',
    updated: '2026-07-23',
    topics: ['ASR', 'Data Augmentation'],
    categories: ['cs.CL', 'eess.AS'],
    status: 'revised',
    paperUrl: 'https://arxiv.org/abs/2607.18064',
    pdfUrl: 'https://arxiv.org/pdf/2607.18064',
  },
  {
    id: '2607.17766',
    title: 'When to Use Extra Context: Evidence-Grounded Terminology Adaptation for Simultaneous Speech Translation',
    abstract: 'A context-selection method for simultaneous speech translation that retrieves terminology only when the acoustic and textual evidence supports intervention.',
    authors: ['Zeyu Yang', 'Marta Ruiz', 'Jonas Pfeiffer'],
    published: '2026-07-20',
    updated: '2026-07-20',
    topics: ['ASR', 'Machine Translation'],
    categories: ['cs.CL'],
    status: 'new',
    codeUrl: 'https://github.com/example/evidence-context',
    paperUrl: 'https://arxiv.org/abs/2607.17766',
    pdfUrl: 'https://arxiv.org/pdf/2607.17766',
  },
  {
    id: '2607.17164',
    title: 'Robust Assamese Speech Recognition through Controlled Fine-Tuning of Whisper Models',
    abstract: 'Controlled fine-tuning experiments isolate the data, normalization, and decoding choices that matter most for low-resource Assamese recognition.',
    authors: ['Ganapati Das', 'Meera Sharma'],
    published: '2026-07-19',
    updated: '2026-07-19',
    topics: ['ASR'],
    categories: ['cs.CL', 'eess.AS'],
    status: 'new',
    paperUrl: 'https://arxiv.org/abs/2607.17164',
    pdfUrl: 'https://arxiv.org/pdf/2607.17164',
  },
  {
    id: '2607.16085',
    title: 'Controlling Implicit Shortcut Reliance in L2 Spoken English Auto-markers',
    abstract: 'The work diagnoses demographic and acoustic shortcuts in spoken-language assessment and introduces controls that improve calibration across learner groups.',
    authors: ['Shilin Gao', 'Rebecca Morris', 'Daniel Ortega'],
    published: '2026-07-17',
    updated: '2026-07-18',
    topics: ['ASR', 'Synthetic Generation'],
    categories: ['cs.CL', 'cs.LG'],
    status: 'revised',
    paperUrl: 'https://arxiv.org/abs/2607.16085',
    pdfUrl: 'https://arxiv.org/pdf/2607.16085',
  },
  {
    id: '2607.13013',
    title: 'Audio-Native Speech Recognition with a Frozen Discrete-Diffusion Language Model',
    abstract: 'A frozen discrete-diffusion language model is connected to an audio encoder, testing whether text-generation priors can support accurate transcription without full language-model adaptation.',
    authors: ['Harsha Vardhan Khurdula', 'Lina Chen'],
    published: '2026-07-14',
    updated: '2026-07-16',
    topics: ['ASR', 'Small Language Models'],
    categories: ['cs.CL', 'cs.AI'],
    status: 'revised',
    codeUrl: 'https://github.com/example/audio-native-ddlm',
    paperUrl: 'https://arxiv.org/abs/2607.13013',
    pdfUrl: 'https://arxiv.org/pdf/2607.13013',
  },
  {
    id: '2607.12468',
    title: 'An Omnilingual-ASR-Based Speech-LLM System for the 2nd MLC-SLM Challenge',
    abstract: 'A multilingual speech-language system combines broad recognition coverage with a compact language model for instruction following under constrained compute.',
    authors: ['Shuming Fang', 'Ravi Menon', 'Elena Petrova'],
    published: '2026-07-14',
    updated: '2026-07-14',
    topics: ['ASR', 'Small Language Models'],
    categories: ['cs.CL', 'eess.AS'],
    status: 'new',
    paperUrl: 'https://arxiv.org/abs/2607.12468',
    pdfUrl: 'https://arxiv.org/pdf/2607.12468',
  },
];

export function searchPreviewPapers(query: string, topic: string): PaperPreview[] {
  const normalizedQuery = query.trim().toLocaleLowerCase();
  return previewPapers.filter((paper) => {
    const matchesTopic = !topic || paper.topics.includes(topic);
    const searchable = [paper.id, paper.title, paper.abstract, ...paper.authors, ...paper.topics, ...paper.categories]
      .join(' ')
      .toLocaleLowerCase();
    return matchesTopic && (!normalizedQuery || searchable.includes(normalizedQuery));
  });
}