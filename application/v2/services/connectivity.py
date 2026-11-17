"""Conectividad y reparación procedentes del Experimento 3."""
from collections import deque
import numpy as np

def componentes(collision):
    free=collision==0; h,w=free.shape; seen=np.zeros((h,w),bool); comps=[]
    for y,x in zip(*np.where(free)):
        if seen[y,x]: continue
        q=deque([(int(y),int(x))]); seen[y,x]=True; comp=[]
        while q:
            cy,cx=q.popleft();comp.append((cy,cx))
            for ny,nx in ((cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)):
                if 0<=ny<h and 0<=nx<w and free[ny,nx] and not seen[ny,nx]:
                    seen[ny,nx]=True;q.append((ny,nx))
        comps.append(comp)
    return sorted(comps,key=len,reverse=True)

def reparar_conectividad(m):
    # Une cada componente secundaria a la principal usando 0-1 BFS.
    t,o,c=(m[k].copy() for k in ('terrain','objects','collision'))
    comps=componentes(c)
    if len(comps)<=1: return dict(m),0
    h,w=c.shape; principal=set(comps[0]); despejadas=set()
    for comp in comps[1:]:
        dist=np.full((h,w),10**9,dtype=np.int32); parent={}; q=deque()
        for y,x in comp:
            dist[y,x]=0;q.appendleft((y,x))
        destino=None
        while q:
            y,x=q.popleft()
            if (y,x) in principal: destino=(y,x);break
            for ny,nx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                if not (0<=ny<h and 0<=nx<w):continue
                # Bordes exteriores de mazmorra no se modifican; aquí se usa solo en bosque.
                coste=int(c[ny,nx]!=0)
                nd=dist[y,x]+coste
                if nd<dist[ny,nx]:
                    dist[ny,nx]=nd;parent[(ny,nx)]=(y,x)
                    (q.appendleft if coste==0 else q.append)((ny,nx))
        if destino is None: continue
        path=[];p=destino
        while p not in comp:
            path.append(p);p=parent[p]
        path.append(p)
        for y,x in path:
            if c[y,x]:despejadas.add((y,x))
            c[y,x]=0;o[y,x]=0;t[y,x]=3
        principal.update(comp);principal.update(path)
    nuevo={**m,'terrain':t,'objects':o,'collision':c}
    return nuevo,len(despejadas)
