"""Select only actual, nearly colocated, level observations of eligible headings."""
def angular_error(a,b):
    return abs((a-b+6)%12-6)


def sector_trace(actions):
    yaw=0
    bins=[]
    for a in actions:
        yaw=(yaw+int(a==1)-int(a==2))%12
        bins.append(yaw)
    return [bins[len(bins)//2-1],bins[-1]]


def covered_pairs(pairs,bins):
    use=[]
    for pair in pairs:
        required=sector_trace(pair["ref"])+sector_trace(pair["alt"])
        if bins and all(min(angular_error(a,b) for b in bins)<=1 for a in required):
            use.append(pair)
    return use


def choose_mode(pairs, observations, pitch):
    """Prefer the existing supported selector. Back off to real contiguous history."""
    info={i[0]:i[1] for i in observations}
    full=covered_pairs(pairs,sorted(info)) if pitch in (0,1) and len(info)>=3 else []
    if full:
        return "full",full,sorted(info)
    if pitch not in (0,1) or 0 not in info or not pairs:
        return "bypass",[],[]
    def chain(sign):
        bins=[0]
        for offset in (1,2):
            b=(sign*offset)%12
            if b not in info:break
            bins.append(b)
        return bins
    # This tie break considers frame recency, never images, labels or route scores.
    chosen=max([chain(1),chain(-1)],key=lambda bins:(len(bins),sum(info[b] for b in bins)))
    return "sparse",pairs,sorted(chosen)
