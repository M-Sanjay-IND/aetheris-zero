'use client';

import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { TelemetryFrame } from '@/hooks/useSimulationStream';

interface BuildingSceneProps {
  telemetry: TelemetryFrame | null;
}

export default function BuildingScene({ telemetry }: BuildingSceneProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const zoneMeshesRef = useRef<THREE.Mesh[]>([]);
  const flowParticlesRef = useRef<THREE.Points | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const container = containerRef.current;

    const width = container.clientWidth || 800;
    const height = container.clientHeight || 500;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.innerHTML = '';
    container.appendChild(renderer.domElement);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0x384358, 1.8);
    scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffa586, 1.5);
    directionalLight.position.set(5, 10, 7);
    scene.add(directionalLight);

    // Building Group
    const building = new THREE.Group();
    scene.add(building);

    // Base glass envelope & wireframe
    const glassMaterial = new THREE.MeshPhongMaterial({
      color: 0x0e1323,
      transparent: true,
      opacity: 0.2,
      shininess: 90,
    });
    const wireframeMaterial = new THREE.MeshBasicMaterial({
      color: 0xb51a2b,
      wireframe: true,
      transparent: true,
      opacity: 0.35,
    });

    const baseGeom = new THREE.BoxGeometry(4, 8, 4);
    const baseMesh = new THREE.Mesh(baseGeom, glassMaterial);
    const wireMesh = new THREE.Mesh(baseGeom, wireframeMaterial);
    building.add(baseMesh);
    building.add(wireMesh);

    // 5 Thermal Zones
    const zones: THREE.Mesh[] = [];
    const defaultPalette = [0xb51a2b, 0xffa586, 0x541a2b, 0x384358, 0x242f49];

    for (let i = 0; i < 5; i++) {
      const zoneGeom = new THREE.BoxGeometry(4.1, 1.5, 4.1);
      const zoneMat = new THREE.MeshPhongMaterial({
        color: defaultPalette[i % defaultPalette.length],
        transparent: true,
        opacity: 0.35,
        emissive: defaultPalette[i % defaultPalette.length],
        emissiveIntensity: 0.25,
      });
      const zone = new THREE.Mesh(zoneGeom, zoneMat);
      zone.position.y = -3.2 + i * 1.6;
      building.add(zone);
      zones.push(zone);
    }
    zoneMeshesRef.current = zones;

    // Airflow Particle System
    const particleCount = 140;
    const particles = new THREE.BufferGeometry();
    const pPositions = new Float32Array(particleCount * 3);
    for (let i = 0; i < particleCount; i++) {
      pPositions[i * 3] = (Math.random() - 0.5) * 4;
      pPositions[i * 3 + 1] = (Math.random() - 0.5) * 8;
      pPositions[i * 3 + 2] = (Math.random() - 0.5) * 4;
    }
    particles.setAttribute('position', new THREE.BufferAttribute(pPositions, 3));
    const pMaterial = new THREE.PointsMaterial({
      color: 0xffa586,
      size: 0.07,
      transparent: true,
      opacity: 0.7,
    });
    const flowParticles = new THREE.Points(particles, pMaterial);
    building.add(flowParticles);
    flowParticlesRef.current = flowParticles;

    camera.position.z = 10;
    camera.position.y = 1;

    let animationFrameId: number;
    let time = 0;

    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);
      time += 0.01;
      building.rotation.y += 0.003;

      // Pulse flow particles
      const positions = flowParticles.geometry.attributes.position.array as Float32Array;
      for (let i = 0; i < particleCount; i++) {
        positions[i * 3 + 1] += 0.02;
        if (positions[i * 3 + 1] > 4) positions[i * 3 + 1] = -4;
      }
      flowParticles.geometry.attributes.position.needsUpdate = true;

      renderer.render(scene, camera);
    };
    animate();

    const handleResize = () => {
      if (!container) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(animationFrameId);
      renderer.dispose();
    };
  }, []);

  // Update zone mesh colors dynamically based on telemetry
  useEffect(() => {
    if (!telemetry || !telemetry.zones) return;
    const zoneKeys = Object.keys(telemetry.zones);
    zoneMeshesRef.current.forEach((mesh, idx) => {
      const zKey = zoneKeys[idx] || `zone_${idx + 1}`;
      const zState = telemetry.zones[zKey];
      if (zState) {
        const hex = zState.hex_color || (zState.temp_c > 24.5 ? '#b51a2b' : zState.temp_c < 20.5 ? '#06b6d4' : '#ffa586');
        const color = new THREE.Color(hex);
        (mesh.material as THREE.MeshPhongMaterial).color = color;
        (mesh.material as THREE.MeshPhongMaterial).emissive = color;
        (mesh.material as THREE.MeshPhongMaterial).emissiveIntensity = (zState.heat_intensity || 0.5) * 0.4;
      }
    });
  }, [telemetry]);

  return (
    <div className="relative w-full h-full flex-grow">
      {/* 3D Canvas Mount Point */}
      <div ref={containerRef} className="w-full h-full" />
    </div>
  );
}

