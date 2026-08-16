import { MapContainer, TileLayer, Marker, Popup, Polygon, useMapEvents } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { useMemo, useRef, useCallback, useState, useEffect } from 'react';
import L from 'leaflet';
import Supercluster from 'supercluster';
import { useGetCasesQuery } from '../features/cases/casesApi';
import { useAppDispatch, useAppSelector } from '../store/hooks';
import { selectFilters, setSelectedCase, setCaseFormOpen, setNewCaseLocation } from '../features/cases/casesSlice';
import { getConvexHull } from '../utils/convexHull';
import { CaseMarker } from './CaseMarker';

const DEFAULT_CENTER: [number, number] = [7.008, 100.472];


const PRIORITY_COLORS: Record<string, string> = {
  P1: '#ef4444',
  P2: '#f97316',
  P3: '#0ea5e9',
};

const RESOURCE_EMOJI: Record<string, string> = {
  medical_evac: '⚕️',
  food_drop: '🍚',
  rescue_boat: '🚤',
  body_recovery: '🆘',
  power_supply: '🔋',
};

const PRIORITY_TEXT: Record<string, string> = {
  P1: 'ด่วนที่สุด (P1)',
  P2: 'ด่วนมาก (P2)',
  P3: 'เฝ้าระวัง (P3)',
};

const RESOURCE_TEXT: Record<string, string> = {
  medical_evac: 'ต้องการแพทย์/พยาบาล',
  food_drop: 'ต้องการอาหารและน้ำ',
  rescue_boat: 'ต้องการเรือกู้ภัยหรือขนย้าย',
  body_recovery: 'ค้นหาผู้สูญหาย/ช่วยชีวิต',
  power_supply: 'ไฟฟ้าชั่วคราว/พลังงาน',
};

const CLUSTER_BUCKETS = [
  { max: 10, color: '#A3E635', label: '1 - 10 เคส' },
  { max: 50, color: '#FDE047', label: '11 - 50 เคส' },
  { max: 200, color: '#FDBA74', label: '51 - 200 เคส' },
  { max: Infinity, color: '#F87171', label: '> 200 เคส' },
];

const iconCache = new Map<string, L.DivIcon>();
const clusterIconCache = new Map<string, L.DivIcon>();

const getResourceEmoji = (tags?: string[]) => {
  if (!tags) return '';
  for (const tag of tags) {
    if (RESOURCE_EMOJI[tag]) return RESOURCE_EMOJI[tag];
  }
  return '';
};

export const createMarkerIcon = (priority?: string) => {
  const color = PRIORITY_COLORS[priority ?? ''] ?? '#0ea5e9';
  const cacheKey = `case-${priority ?? 'NA'}`;
  if (!iconCache.has(cacheKey)) {
    iconCache.set(
      cacheKey,
      L.divIcon({
        className: 'case-marker-icon',
        html: `<div class="case-dot" data-priority="${priority ?? 'P3'}"></div>`,
        iconSize: [26, 26],
        iconAnchor: [13, 13],
        popupAnchor: [0, -10],
      })
    );
  }
  return iconCache.get(cacheKey)!;
};

const priorityLabel = (code?: string) => {
  if (code === 'P1') return 'P1 • ด่วนที่สุด';
  if (code === 'P2') return 'P2 • ด่วนมาก';
  if (code === 'P3') return 'P3 • เฝ้าระวัง';
  return code ?? 'ไม่ระบุ';
};

const describeResourceTag = (tag: string) => {
  const label = RESOURCE_TEXT[tag];
  if (!label) return undefined;
  const emoji = RESOURCE_EMOJI[tag];
  return emoji ? `${emoji} ${label}` : label;
};

const pickClusterColor = (count: number) => {
  const bucket = CLUSTER_BUCKETS.find((step) => count <= step.max);
  return bucket?.color ?? '#94a3b8';
};

const createClusterIcon = (count: number) => {
  const color = pickClusterColor(count);
  const cacheKey = `cluster-${color}-${Math.min(9999, count)}`;
  if (!clusterIconCache.has(cacheKey)) {
    const size = count > 500 ? 64 : count > 200 ? 56 : count > 50 ? 48 : 40;
    clusterIconCache.set(
      cacheKey,
      L.divIcon({
        className: 'cluster-marker__wrapper',
        html: `
          <div class="cluster-marker" style="background:${color}; width:${size}px; height:${size}px">
            <span>${count}</span>
          </div>
        `,
        iconSize: [size, size],
        iconAnchor: [size / 2, size / 2],
      })
    );
  }
  return clusterIconCache.get(cacheKey)!;
};

export type CasePointProps = {
  cluster: false;
  caseId: string;
  priority: string;
  status: string;
  text: string;
  resourceTags: string[];
  needs?: string;
  lat: number;
  lng: number;
};

type ClusterProps = {
  cluster: true;
  point_count: number;
  point_count_abbreviated: number;
};

interface MapControllerProps {
  onBoundsChange: (bounds: [number, number, number, number]) => void;
  onZoomChange: (zoom: number) => void;
  onMapClick: (lat: number, lng: number) => void;
}

function MapController({ onBoundsChange, onZoomChange, onMapClick }: MapControllerProps) {
  const map = useMapEvents({
    moveend: () => {
      const bounds = map.getBounds();
      onBoundsChange([bounds.getWest(), bounds.getSouth(), bounds.getEast(), bounds.getNorth()]);
      onZoomChange(map.getZoom());
    },
    zoomend: () => {
      const bounds = map.getBounds();
      onBoundsChange([bounds.getWest(), bounds.getSouth(), bounds.getEast(), bounds.getNorth()]);
      onZoomChange(map.getZoom());
    },
    click: (e) => {
      onMapClick(e.latlng.lat, e.latlng.lng);
    }
  });

  useEffect(() => {
    const bounds = map.getBounds();
    onBoundsChange([bounds.getWest(), bounds.getSouth(), bounds.getEast(), bounds.getNorth()]);
    onZoomChange(map.getZoom());
  }, [map, onBoundsChange, onZoomChange]);

  return null;
}

export function MapView() {
  const filters = useAppSelector(selectFilters);
  const dispatch = useAppDispatch();
  const { data: cases = [] } = useGetCasesQuery({ ...filters, mode: 'map' });
  const mapRef = useRef<L.Map | null>(null);
  const [mapBounds, setMapBounds] = useState<[number, number, number, number]>([
    -180, -85, 180, 85,
  ]);
  const [mapZoom, setMapZoom] = useState(11);
  const [highlightPolygon, setHighlightPolygon] = useState<[number, number][] | null>(null);

  const casePoints = useMemo<Supercluster.PointFeature<CasePointProps>[]>(() => {
    return cases
      .filter((item) => Array.isArray(item.location?.coordinates) && item.location?.coordinates.length === 2)
      .map((item) => {
        const [lng, lat] = item.location!.coordinates as [number, number];
        return {
          type: 'Feature' as const,
          properties: {
            cluster: false,
            caseId: item._id,
            priority: item.priority_label,
            status: item.status,
            text: item.text.slice(0, 200),
            resourceTags: item.resource_tags ?? [],
            needs: item.resource_tags
              ?.map((tag) => describeResourceTag(tag))
              .filter(Boolean)
              .join(' • '),
            lat,
            lng,
          },
          geometry: {
            type: 'Point' as const,
            coordinates: [lng, lat] as [number, number],
          },
        };
      });
  }, [cases]);

  const clusterIndex = useMemo(() => {
    const instance = new Supercluster<CasePointProps, ClusterProps>({
      radius: 100,
      maxZoom: 15,
      minZoom: 0,
      minPoints: 2,
    });
    instance.load(casePoints);
    return instance;
  }, [casePoints]);

  const clusters = useMemo<
    Array<Supercluster.PointFeature<CasePointProps> | Supercluster.ClusterFeature<ClusterProps>>
  >(() => {
    return clusterIndex.getClusters(mapBounds, Math.round(mapZoom));
  }, [clusterIndex, mapBounds, mapZoom]);



  const handleMapClick = useCallback((lat: number, lng: number) => {
    dispatch(setNewCaseLocation({ lat, lng }));
    dispatch(setCaseFormOpen(true));
  }, [dispatch]);

  const getClusterLeaves = useCallback(
    (clusterId: number) => {
      return clusterIndex.getLeaves(clusterId, Infinity);
    },
    [clusterIndex]
  );

  const handleClusterClick = useCallback(
    (clusterId: number) => {
      const leaves = getClusterLeaves(clusterId);
      if (leaves.length === 0) return;

      const bounds = L.latLngBounds(leaves.map((l) => [l.geometry.coordinates[1], l.geometry.coordinates[0]]));
      mapRef.current?.fitBounds(bounds, { padding: [50, 50], animate: true });
    },
    [getClusterLeaves]
  );

  const handleClusterHover = useCallback(
    (clusterId: number) => {
      const leaves = getClusterLeaves(clusterId);
      const points: [number, number][] = leaves.map((l) => [l.geometry.coordinates[1], l.geometry.coordinates[0]]);
      const hull = getConvexHull(points);
      setHighlightPolygon(hull);
    },
    [getClusterLeaves]
  );

  useEffect(() => {
    if (mapZoom >= 16) {
      setHighlightPolygon(null);
    }
  }, [mapZoom]);

  const renderCluster = (clusterFeature: Supercluster.ClusterFeature<ClusterProps>) => {
    const [lng, lat] = clusterFeature.geometry.coordinates;
    const count = clusterFeature.properties.point_count;
    const clusterId = Number(clusterFeature.id);
    return (
      <Marker
        key={`cluster-${clusterId}`}
        position={[lat, lng]}
        icon={createClusterIcon(count)}
        eventHandlers={{
          click: () => handleClusterClick(clusterId),
          mouseover: () => handleClusterHover(clusterId),
          mouseout: () => setHighlightPolygon(null),
        }}
      />
    );
  };

  return (
    <div className="map-wrapper">
      <MapContainer
        center={DEFAULT_CENTER}
        zoom={12}
        style={{ width: '100%', height: '100%' }}
        ref={mapRef}
      >
        <MapController 
          onBoundsChange={setMapBounds}
          onZoomChange={setMapZoom}
          onMapClick={handleMapClick}
        />
        <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
        {highlightPolygon && (
          <Polygon
            positions={highlightPolygon}
            pathOptions={{
              color: '#3b82f6',
              weight: 2,
              opacity: 0.8,
              fillOpacity: 0.15,
              dashArray: '5, 5',
            }}
          />
        )}
        {clusters.map((cluster) =>
          cluster.properties.cluster
            ? renderCluster(cluster as Supercluster.ClusterFeature<ClusterProps>)
            : <CaseMarker 
                key={(cluster as Supercluster.PointFeature<CasePointProps>).properties.caseId}
                feature={cluster as Supercluster.PointFeature<CasePointProps>}
                createMarkerIcon={createMarkerIcon}
              />
        )}
      </MapContainer>
      <div className="map-legend">
        <div className="map-legend__group">
          <p>ระดับความเร่งด่วน</p>
          <ul>
            {Object.entries(PRIORITY_COLORS).map(([code, color]) => (
              <li key={code}>
                <span className="legend-dot" style={{ background: color }} />
                {PRIORITY_TEXT[code]}
              </li>
            ))}
          </ul>
        </div>
        <div className="map-legend__group">
          <p>ความต้องการสำคัญ</p>
          <ul>
            {Object.entries(RESOURCE_EMOJI).map(([key, emoji]) => (
              <li key={key}>
                <span className="legend-emoji">{emoji}</span>
                {RESOURCE_TEXT[key]}
              </li>
            ))}
          </ul>
        </div>
        <div className="map-legend__group">
          <p>จำนวนเคส/ความหนาแน่น</p>
          <ul>
            {CLUSTER_BUCKETS.map((bucket) => (
              <li key={bucket.label}>
                <span className="legend-circle" style={{ background: bucket.color }} />
                {bucket.label}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

