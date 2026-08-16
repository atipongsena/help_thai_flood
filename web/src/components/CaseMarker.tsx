import React from 'react';
import { Marker, Popup } from 'react-leaflet';
import Supercluster, { PointFeature } from 'supercluster';
import { CasePointProps } from './MapView';
import { createMarkerIcon } from './MapView';
import { useAppDispatch } from '../store/hooks';
import { setSelectedCase } from '../features/cases/casesSlice';

interface CaseMarkerProps {
  feature: PointFeature<CasePointProps>;
  createMarkerIcon: (priority: string) => L.DivIcon;
}

export const CaseMarker = React.memo(({ feature, createMarkerIcon }: CaseMarkerProps) => {
  const dispatch = useAppDispatch();
  const [lng, lat] = feature.geometry.coordinates;
  const { caseId, priority, status, text, needs } = feature.properties;
  const googleUrl = `https://www.google.com/maps?q=${lat},${lng}`;

  return (
    <Marker
      key={caseId}
      position={[lat, lng]}
      icon={createMarkerIcon(priority)}
    >
      <Popup className="rich-popup">
        <div className="popup-card">
          <div className="popup-header">
            <div className="flex items-center gap-2">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>
                <polyline points="9 22 9 12 15 12 15 22"></polyline>
              </svg>
              <span className="font-bold">สถานะ: {status}</span>
            </div>
          </div>
          
          <div className="popup-body">
            <div className="text-xs text-muted mb-2">ID: {caseId.slice(0, 8)}...</div>
            <p className="text-sm text-gray-700 mb-3 line-clamp-4">{text}</p>
            
            {needs && (
              <div className="needs-tag">
                {needs}
              </div>
            )}
          </div>

          <div className="popup-footer">
            <button
              onClick={() => dispatch(setSelectedCase(caseId))}
              className="view-details-btn"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                <polyline points="14 2 14 8 20 8"></polyline>
                <line x1="16" y1="13" x2="8" y2="13"></line>
                <line x1="16" y1="17" x2="8" y2="17"></line>
                <polyline points="10 9 9 9 8 9"></polyline>
              </svg>
              ดูรายละเอียด
            </button>
            <a 
              href={googleUrl} 
              target="_blank" 
              rel="noreferrer"
              className="map-link"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path>
                <circle cx="12" cy="10" r="3"></circle>
              </svg>
            </a>
          </div>
        </div>
      </Popup>
    </Marker>
  );
});
