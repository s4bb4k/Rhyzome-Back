"""Generador B del Experimento 3 (misma lógica, código extraído)."""
from .semantic import generar_A, WIDTH, HEIGHT
from application.v2.services.connectivity import reparar_conectividad

def generar_B(bioma,seed,h=HEIGHT,w=WIDTH):
    original=generar_A(bioma,seed,h,w)
    m={**original,'terrain':original['terrain'].copy(),
       'objects':original['objects'].copy(),'collision':original['collision'].copy()}
    cambios=0
    if bioma=='bosque':
        m,cambios=reparar_conectividad(m)
    elif bioma=='desierto':
        x=w//2
        m['terrain'][:,x:x+2]=3
        m['objects'][:,x:x+2]=0
        m['collision'][:,x:x+2]=0
    elif bioma=='ciudad':
        cy,cx=h//2,w//2
        # Accesos centrales despejados hacia la plaza, sin edificios ni árboles.
        m['terrain'][cy, :]=5; m['terrain'][:,cx]=5
        m['objects'][cy,:]=0;m['objects'][:,cx]=0
        m['collision'][cy,:]=0;m['collision'][:,cx]=0
    elif bioma=='mazmorra':
        reparado,cambios=reparar_conectividad(m)
        # Conservar suelo de piedra; solo se cambia la ocupación de muros.
        reparado['terrain'][:]=4
        m=reparado
    return m,cambios
