import React from 'react';
import PollingSidebar from '../../../../frontend_components/PollingSidebar';

export default function Home() {
  return (
    <main className="flex min-h-screen items-center justify-center bg-black">
      <div className="flex h-[800px] w-[1200px] border border-slate-700 rounded-xl overflow-hidden bg-slate-900 shadow-2xl">
        <div className="flex-1 flex items-center justify-center p-8 bg-slate-950">
          <div className="text-center">
            <h1 className="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-sky-400 to-emerald-400 mb-4">
              AI Detection Live Feed
            </h1>
            <p className="text-slate-400">
              The camera feed should appear here, while detections stream into the sidebar.
            </p>
          </div>
        </div>
        <PollingSidebar />
      </div>
    </main>
  );
}
