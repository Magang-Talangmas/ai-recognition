import React, { useEffect, useState } from 'react';

// Define TypeScript interfaces for the detection data
interface BBox {
  0: number; // x1
  1: number; // y1
  2: number; // x2
  3: number; // y2
}

interface DetectionDetail {
  name: string;
  similarity: number;
  bbox: BBox | number[];
}

interface DetectionPayload {
  active_names: string[];
  details: DetectionDetail[];
}

const PollingSidebar: React.FC = () => {
  const [detections, setDetections] = useState<DetectionPayload>({
    active_names: [],
    details: [],
  });
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    // Fetch data via HTTP GET every 1000ms
    const fetchDetections = async () => {
      try {
        const response = await fetch('http://192.168.77.171:8088/detect');
        if (response.ok) {
          const data: DetectionPayload = await response.json();
          setDetections(data);
          setIsConnected(true);
        } else {
          setIsConnected(false);
        }
      } catch (error) {
        console.error('[Polling] Error fetching detection data:', error);
        setIsConnected(false);
      }
    };

    // Initial fetch
    fetchDetections();

    // Set up polling interval
    const intervalId = setInterval(fetchDetections, 1000);

    // Cleanup on unmount
    return () => {
      clearInterval(intervalId);
    };
  }, []);

  return (
    <div className="w-64 h-full bg-slate-900 text-slate-100 p-4 shadow-xl flex flex-col border-r border-slate-700">
      <h2 className="text-xl font-bold text-sky-400 mb-4 flex items-center justify-between">
        Live Detections
        <span className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} title={isConnected ? "Connected" : "Disconnected"}></span>
      </h2>
      
      <div className="flex-1 overflow-y-auto">
        {detections.active_names.length === 0 ? (
          <p className="text-slate-400 text-sm italic">No one detected</p>
        ) : (
          <ul className="space-y-3">
            {detections.details.map((detail, index) => (
              <li key={`${detail.name}-${index}`} className="bg-slate-800 p-3 rounded-lg border border-slate-600 shadow-sm">
                <div className="font-semibold text-white">{detail.name}</div>
                <div className="text-xs text-slate-400 mt-1 flex justify-between">
                  <span>Sim: {(detail.similarity).toFixed(1)}</span>
                  <span title="Bounding Box">[x: {detail.bbox[0]}, y: {detail.bbox[1]}]</span>
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
};

export default PollingSidebar;
