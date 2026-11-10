"""Generadores A de Experimento 2/3, aislados de los generadores V1.
IMPORTANTE: ruido_suave es interpolación bilineal; no equivale a Perlin V1.
"""
import numpy as np

TERRAIN = {'cesped':0, 'agua':1, 'arena':2, 'camino':3, 'piedra':4, 'calle':5, 'plaza':6}
OBJECT = {'vacio':0, 'arbol':1, 'roca':2, 'cactus':3, 'edificio':4, 'muro':5, 'puerta':6, 'decoracion':7}
BIOMAS = ('bosque', 'desierto', 'ciudad', 'mazmorra')
WIDTH, HEIGHT = 50, 35

def ruido_suave(rng, h, w, escalas=(3, 7, 15)):
    # Interpolación bilineal de pequeñas rejillas aleatorias, ponderadas por escala.
    yy, xx = np.mgrid[0:h, 0:w]
    total = np.zeros((h,w), float); suma = 0.
    for n, peso in zip(escalas, (0.55,0.30,0.15)):
        gh = max(2, int(np.ceil(h/n))+1); gw = max(2,int(np.ceil(w/n))+1)
        coarse = rng.random((gh,gw))
        yf = yy*(gh-1)/max(1,h-1); xf = xx*(gw-1)/max(1,w-1)
        y0 = np.minimum(yf.astype(int),gh-2); x0 = np.minimum(xf.astype(int),gw-2)
        a=yf-y0; b=xf-x0
        interp=(1-a)*(1-b)*coarse[y0,x0]+a*(1-b)*coarse[y0+1,x0]+(1-a)*b*coarse[y0,x0+1]+a*b*coarse[y0+1,x0+1]
        total += peso*interp; suma += peso
    total /= suma
    return (total-total.min())/max(1e-12,total.max()-total.min())

def cellular(rng,h,w,prob=0.43,iterations=4):
    walls = rng.random((h,w)) < prob
    walls[[0,-1],:] = True; walls[:,[0,-1]] = True
    for _ in range(iterations):
        padded=np.pad(walls.astype(int),1,constant_values=1)
        count=sum(padded[1+dy:h+1+dy,1+dx:w+1+dx]
                  for dy in (-1,0,1) for dx in (-1,0,1) if (dy,dx)!=(0,0))
        walls = np.where(walls,count>=4,count>=5)
        walls[[0,-1],:] = True; walls[:,[0,-1]] = True
    return walls

def generar_A(bioma, seed, h=HEIGHT, w=WIDTH):
    rng=np.random.default_rng(int(seed))
    terrain=np.zeros((h,w), dtype=np.uint8)
    objects=np.zeros((h,w), dtype=np.uint8)
    if bioma=='bosque':
        noise=ruido_suave(rng,h,w)
        terrain[noise<0.17]=1
        cy=h//2
        for x in range(w):
            y=int(np.clip(cy+3*np.sin(x/7+seed%17),1,h-2))
            terrain[max(0,y-1):min(h,y+2),x]=3
        mask=(terrain==0)&(rng.random((h,w))<0.31)
        objects[mask]=1
        mask=(terrain==0)&(objects==0)&(rng.random((h,w))<0.035)
        objects[mask]=2
    elif bioma=='desierto':
        terrain[:]=2
        noise=ruido_suave(rng,h,w)
        terrain[noise<0.12]=1
        y=h//2
        terrain[y:y+2,:]=3
        mask=(terrain==2)&(rng.random((h,w))<0.09)
        objects[mask]=3
        mask=(terrain==2)&(objects==0)&(rng.random((h,w))<0.04)
        objects[mask]=2
    elif bioma=='ciudad':
        terrain[:]=0
        for x in range(5,w,10): terrain[:,x:x+2]=5
        for y in range(5,h,9): terrain[y:y+2,:]=5
        cy,cx=h//2,w//2
        terrain[cy-3:cy+4,cx-4:cx+5]=6
        # Los edificios no ocupan calles ni plaza; algunos espacios verdes quedan libres.
        for y in range(2,h-2,3):
            for x in range(2,w-2,4):
                area=terrain[y:y+2,x:x+3]
                if area.shape==(2,3) and np.all(area==0) and rng.random()<0.73:
                    objects[y:y+2,x:x+3]=4
        mask=(terrain==0)&(objects==0)&(rng.random((h,w))<0.025)
        objects[mask]=1
    elif bioma=='mazmorra':
        terrain[:]=4
        walls=cellular(rng,h,w)
        objects[walls]=5
        # Garantiza un eje transitable continuo, sin destruir todo el diseño celular.
        y=h//2; objects[y-1:y+2,1:w-1]=0
        # Dos puertas decorativas atravesables en el corredor.
        for x in (w//3,2*w//3): objects[y,x]=6
        mask=(objects==0)&(rng.random((h,w))<0.02)
        objects[mask]=7
    else:
        raise ValueError('Bioma no reconocido: '+str(bioma))
    collision=np.isin(terrain,[1]).astype(np.uint8)
    collision[np.isin(objects,[1,2,3,4,5])]=1
    collision[np.isin(objects,[6,7])]=0
    return {'bioma':bioma,'seed':int(seed),'terrain':terrain,'objects':objects,'collision':collision}
