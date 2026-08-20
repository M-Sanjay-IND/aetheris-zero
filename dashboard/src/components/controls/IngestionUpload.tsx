'use client';

import React, { useState } from 'react';

interface IngestionUploadProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function IngestionUpload({ isOpen, onClose }: IngestionUploadProps) {
  const [tagsInput, setTagsInput] = useState('AHU1_SAT, VAV101_ZN_T, CHW_SUP_T, VAV102_DMPR_POS');
  const [parsedResults, setParsedResults] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [turtleGraph, setTurtleGraph] = useState<string>('');

  if (!isOpen) return null;

  const handleParse = async () => {
    setLoading(true);
    try {
      const tags = tagsInput.split(',').map((t) => t.trim()).filter(Boolean);
      const res = await fetch('http://localhost:8000/api/v1/ingestion/parse-tags', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tags }),
      });
      const data = await res.json();
      setParsedResults(data.points || []);

      const ttlRes = await fetch('http://localhost:8000/api/v1/ingestion/graph-turtle');
      const ttlData = await ttlRes.json();
      setTurtleGraph(ttlData.turtle || '');
    } catch (e) {
      console.error('Failed to parse tags', e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-background/80 backdrop-blur-md flex items-center justify-center p-4">
      <div className="bg-surface-container-high border border-grid-line rounded-lg w-full max-w-2xl max-h-[85vh] flex flex-col overflow-hidden shadow-2xl">
        {/* Header */}
        <div className="p-4 border-b border-grid-line flex justify-between items-center bg-surface-container">
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-primary-fixed">schema</span>
            <h3 className="font-headline-md text-base text-on-surface">Semantic Ingestion & Brick Schema v1.3</h3>
          </div>
          <button onClick={onClose} className="text-on-surface-variant hover:text-on-surface">
            <span className="material-symbols-outlined">close</span>
          </button>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto flex flex-col gap-4">
          <div>
            <label className="block font-label-caps text-xs text-on-surface-variant uppercase mb-1">
              Zero-Shot BACnet / Modbus Tag Tokenizer
            </label>
            <input
              type="text"
              value={tagsInput}
              onChange={(e) => setTagsInput(e.target.value)}
              className="w-full bg-surface-container-lowest border border-grid-line rounded p-2 text-xs font-data-telemetry text-primary-fixed focus:outline-none focus:border-primary-fixed"
              placeholder="Comma separated BACnet tags..."
            />
          </div>

          <button
            onClick={handleParse}
            disabled={loading}
            className="w-full py-2 bg-primary-fixed text-on-primary font-label-caps text-xs uppercase font-bold rounded hover:bg-primary-fixed/90 transition-colors"
          >
            {loading ? 'Classifying Entities...' : 'Parse Tags & Extract Brick Priors'}
          </button>

          {parsedResults.length > 0 && (
            <div>
              <h4 className="font-label-caps text-xs text-on-surface-variant uppercase mb-2">Classified Entities</h4>
              <div className="bg-surface-container-lowest border border-grid-line rounded p-3 max-h-40 overflow-y-auto flex flex-col gap-1 text-xs font-data-telemetry">
                {parsedResults.map((p, i) => (
                  <div key={i} className="flex justify-between border-b border-grid-line/30 pb-1">
                    <span className="text-primary-fixed">{p.raw_tag}</span>
                    <span className="text-on-surface-variant">{p.brick_class}</span>
                    <span className="text-secondary">{p.equipment_id}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {turtleGraph && (
            <div>
              <h4 className="font-label-caps text-xs text-on-surface-variant uppercase mb-2">Brick Schema Turtle (RDF Graph)</h4>
              <pre className="bg-surface-container-lowest border border-grid-line rounded p-3 max-h-48 overflow-y-auto text-[11px] font-data-telemetry text-on-surface-variant">
                {turtleGraph}
              </pre>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

