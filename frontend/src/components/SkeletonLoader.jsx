export default function SkeletonLoader({ className = '', style = {} }) {
  return (
    <div className={`rounded-xl overflow-hidden ${className}`} style={{ width: 200, ...style }}>
      {/* Poster skeleton */}
      <div className="skeleton h-48 w-full" />
      {/* Info skeleton */}
      <div className="bg-[#111] p-3 space-y-2">
        <div className="skeleton h-4 rounded w-4/5" />
        <div className="skeleton h-3 rounded w-3/5" />
        <div className="flex gap-1.5 pt-1">
          <div className="skeleton h-4 w-14 rounded-full" />
          <div className="skeleton h-4 w-10 rounded-full" />
        </div>
      </div>
    </div>
  )
}

export function SkeletonText({ lines = 3, className = '' }) {
  return (
    <div className={`space-y-2 ${className}`}>
      {Array.from({ length: lines }).map((_, i) => (
        <div
          key={i}
          className="skeleton h-4 rounded"
          style={{ width: `${100 - i * 15}%` }}
        />
      ))}
    </div>
  )
}

export function SkeletonRow() {
  return (
    <section className="py-6">
      <div className="px-4 sm:px-8 mb-4">
        <div className="skeleton h-7 w-48 rounded" />
      </div>
      <div className="flex gap-4 px-4 sm:px-8 overflow-hidden">
        {Array.from({ length: 6 }).map((_, i) => (
          <SkeletonLoader key={i} />
        ))}
      </div>
    </section>
  )
}
