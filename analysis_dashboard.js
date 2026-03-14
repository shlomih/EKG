import React from 'react';

const DashboardMock = () => {
  const safetyFlags = [
    { label: "Pacemaker/ICD", active: false, status: "CLEAR" },
    { label: "Athlete Mode", active: true, status: "ACTIVE" },
    { label: "Pregnancy", active: false, status: "N/A" },
  ];

  return (
    <div style={{ background: '#051211', color: '#E0F7F4', padding: '2rem', fontFamily: 'sans-serif', minHeight: '100vh' }}>
      {/* 1. Urgent Alert Bar */}
      <div style={{ background: '#FF6B6B22', border: '1px solid #FF6B6B', padding: '1rem', borderRadius: '8px', marginBottom: '2rem' }}>
        <h3 style={{ color: '#FF6B6B', margin: 0 }}>⚠️ Potential OMI Detected</h3>
        <p style={{ margin: '5px 0 0 0', opacity: 0.8, fontSize: '0.9rem' }}>Subtle T-wave peaking in Leads V2-V4. Recommendation: Emergent Cardiology Consultation.</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 300px', gap: '2rem' }}>

        {/* 2. Main EKG Viewport */}
        <section>
          <div style={{ border: '1px solid #00E5B033', borderRadius: '12px', padding: '1.5rem', background: '#0D1F1E' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1rem' }}>
              <span style={{ fontFamily: 'DM Mono', color: '#00E5B0' }}>LEAD II - PROCESSED</span>
              <span style={{ fontSize: '0.8rem', opacity: 0.6 }}>Digitized from: Paper Scan (High Res)</span>
            </div>

            {/* Mock Waveform (Placeholder for Recharts) */}
            <div style={{ height: '200px', width: '100%', background: 'linear-gradient(transparent 95%, #00E5B011 95%), linear-gradient(90deg, transparent 95%, #00E5B011 95%)', backgroundSize: '20px 20px', position: 'relative' }}>
               <svg viewBox="0 0 800 200" style={{ width: '100%', height: '100%' }}>
                  <path d="M0,100 L50,100 L60,80 L70,120 L80,100 L120,100 L130,20 L145,180 L160,100 L200,100"
                        fill="none" stroke="#00E5B0" strokeWidth="2" />
                  {/* AI Explanation Saliency Overlay */}
                  <rect x="120" y="20" width="40" height="160" fill="#FF6B6B33" />
               </svg>
            </div>
            <p style={{ fontSize: '0.75rem', marginTop: '1rem', color: '#FF6B6B' }}>Highlighted area indicates segment driving AI suspicion.</p>
          </div>
        </section>

        {/* 3. Clinical Context Panel (Logic Inverters) */}
        <aside>
          <h4 style={{ textTransform: 'uppercase', letterSpacing: '0.1em', fontSize: '0.8rem', marginBottom: '1rem' }}>Logic Inverters</h4>
          {safetyFlags.map(flag => (
            <div key={flag.label} style={{ background: '#142B29', padding: '1rem', borderRadius: '8px', marginBottom: '10px', borderLeft: `4px solid ${flag.active ? '#00E5B0' : '#444'}` }}>
              <div style={{ fontSize: '0.7rem', opacity: 0.6 }}>{flag.label}</div>
              <div style={{ fontWeight: 'bold', color: flag.active ? '#00E5B0' : '#E0F7F4' }}>{flag.status}</div>
            </div>
          ))}

          <div style={{ marginTop: '2rem', padding: '1rem', border: '1px solid #7AA8A444', borderRadius: '8px' }}>
            <h4 style={{ fontSize: '0.8rem', margin: '0 0 10px 0' }}>Patient Stats</h4>
            <div style={{ fontSize: '0.9rem', lineHeight: '1.6' }}>
              67Y Male • 88kg<br/>
              K+: 3.8 mmol/L<br/>
              QTc: 442ms (Normal)
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
};
