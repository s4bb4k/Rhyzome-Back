"""Métricas del Experimento 3, sin redefinir las métricas V1."""
from collections import deque
import numpy as np

def conectividad(collision):
    free=collision==0; h,w=free.shape
    seen=np.zeros((h,w),bool); sizes=[]
    for sy,sx in zip(*np.where(free)):
        if seen[sy,sx]: continue
        seen[sy,sx]=True; q=deque([(int(sy),int(sx))]); size=0
        while q:
            y,x=q.popleft();size+=1
            for ny,nx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                if 0<=ny<h and 0<=nx<w and free[ny,nx] and not seen[ny,nx]:
                    seen[ny,nx]=True;q.append((ny,nx))
        sizes.append(size)
    nfree=int(free.sum())
    return (max(sizes)/nfree if nfree else 0.,len(sizes))

def metricas(m):
    t,o,c=m['terrain'],m['objects'],m['collision']
    ratio,nregions=conectividad(c)
    # No debe haber objeto sólido sobre agua ni edificios sobre calles/plaza.
    invalid=int(((o>0)&(t==1)).sum()+((o==4)&np.isin(t,[5,6])).sum())
    return {'bioma':m['bioma'],'seed':m['seed'],
            'transitabilidad':float((c==0).mean()),'mayor_region_ratio':float(ratio),
            'regiones':int(nregions),'densidad_objetos':float((o>0).mean()),
            'solapamientos_invalidos':invalid,
            'tipos_terreno':int(len(np.unique(t))),
            'tipos_objetos':int(len(np.unique(o[o>0])))}
